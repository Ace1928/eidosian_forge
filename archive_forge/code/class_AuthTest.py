import asyncio
import logging
import os
import shutil
import sys
import warnings
from contextlib import contextmanager
import pytest
import zmq
import zmq.asyncio
import zmq.auth
from zmq.tests import SkipTest, skip_pypy
@pytest.mark.usefixtures('_async_setup')
class AuthTest:
    auth = None

    async def async_setup(self):
        self.context = zmq.asyncio.Context()
        if zmq.zmq_version_info() < (4, 0):
            raise SkipTest('security is new in libzmq 4.0')
        try:
            zmq.curve_keypair()
        except zmq.ZMQError:
            raise SkipTest('security requires libzmq to have curve support')
        logging.getLogger('zmq.auth').setLevel(logging.DEBUG)
        self.auth = self.make_auth()
        await self.start_auth()

    async def async_teardown(self):
        if sys.platform.startswith('win'):
            await asyncio.sleep(0.2)
        if self.auth:
            self.auth.stop()
            self.auth = None
        if sys.platform.startswith('win'):
            await asyncio.sleep(0.2)
        self.context.term()

    def make_auth(self):
        raise NotImplementedError()

    async def start_auth(self):
        self.auth.start()

    async def can_connect(self, server, client, timeout=1000):
        """Check if client can connect to server using tcp transport"""
        result = False
        iface = 'tcp://127.0.0.1'
        port = server.bind_to_random_port(iface)
        client.connect('%s:%i' % (iface, port))
        msg = [b'Hello World']
        await server.poll(100, zmq.POLLOUT)
        if await server.poll(timeout, zmq.POLLOUT):
            try:
                await server.send_multipart(msg, zmq.NOBLOCK)
            except zmq.Again:
                warnings.warn('server set POLLOUT, but cannot send', RuntimeWarning)
                return False
        else:
            return False
        if await client.poll(timeout):
            try:
                rcvd_msg = await client.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                warnings.warn('client set POLLIN, but cannot recv', RuntimeWarning)
            else:
                assert rcvd_msg == msg
                result = True
        return result

    @contextmanager
    def push_pull(self):
        with self.context.socket(zmq.PUSH) as server, self.context.socket(zmq.PULL) as client:
            server.linger = 0
            server.sndtimeo = 2000
            client.linger = 0
            client.rcvtimeo = 2000
            yield (server, client)

    @contextmanager
    def curve_push_pull(self, certs, client_key='ok'):
        server_public, server_secret, client_public, client_secret = certs
        with self.push_pull() as (server, client):
            server.curve_publickey = server_public
            server.curve_secretkey = server_secret
            server.curve_server = True
            if client_key is not None:
                client.curve_publickey = client_public
                client.curve_secretkey = client_secret
                if client_key == 'ok':
                    client.curve_serverkey = server_public
                else:
                    private, public = zmq.curve_keypair()
                    client.curve_serverkey = public
            yield (server, client)

    async def test_null(self):
        """threaded auth - NULL"""
        self.auth.stop()
        self.auth = None
        self.context.term()
        self.context = zmq.asyncio.Context()
        with self.push_pull() as (server, client):
            assert await self.can_connect(server, client)
        with self.push_pull() as (server, client):
            server.zap_domain = b'global'
            assert await self.can_connect(server, client)

    async def test_deny(self):
        self.auth.deny('127.0.0.1')
        with pytest.raises(ValueError):
            self.auth.allow('127.0.0.2')
        with self.push_pull() as (server, client):
            server.zap_domain = b'global'
            assert not await self.can_connect(server, client, timeout=100)

    async def test_allow(self):
        self.auth.allow('127.0.0.1')
        with pytest.raises(ValueError):
            self.auth.deny('127.0.0.2')
        with self.push_pull() as (server, client):
            server.zap_domain = b'global'
            assert await self.can_connect(server, client)

    @pytest.mark.parametrize('enabled, password, success', [(True, 'correct', True), (False, 'correct', False), (True, 'incorrect', False)])
    async def test_plain(self, enabled, password, success):
        """threaded auth - PLAIN"""
        with self.push_pull() as (server, client):
            server.plain_server = True
            if password:
                client.plain_username = b'admin'
                client.plain_password = password.encode('ascii')
            if enabled:
                self.auth.configure_plain(domain='*', passwords={'admin': 'correct'})
            if success:
                assert await self.can_connect(server, client)
            else:
                assert not await self.can_connect(server, client, timeout=100)
        self.auth.stop()
        self.auth = None
        with self.push_pull() as (server, client):
            assert await self.can_connect(server, client)

    @pytest.mark.parametrize('client_key, location, success', [('ok', zmq.auth.CURVE_ALLOW_ANY, True), ('ok', 'public_keys', True), ('bad', 'public_keys', False), (None, 'public_keys', False)])
    async def test_curve(self, certs, public_keys_dir, client_key, location, success):
        """threaded auth - CURVE"""
        self.auth.allow('127.0.0.1')
        with self.curve_push_pull(certs, client_key=client_key) as (server, client):
            if location:
                if location == 'public_keys':
                    location = public_keys_dir
                self.auth.configure_curve(domain='*', location=location)
            if success:
                assert await self.can_connect(server, client, timeout=100)
            else:
                assert not await self.can_connect(server, client, timeout=100)
        self.auth.stop()
        self.auth = None
        with self.push_pull() as (server, client):
            assert await self.can_connect(server, client)

    @pytest.mark.parametrize('key', ['ok', 'wrong'])
    @pytest.mark.parametrize('async_', [True, False])
    async def test_curve_callback(self, certs, key, async_):
        """threaded auth - CURVE with callback authentication"""
        self.auth.allow('127.0.0.1')
        server_public, server_secret, client_public, client_secret = certs

        class CredentialsProvider:

            def __init__(self):
                if key == 'ok':
                    self.client = client_public
                else:
                    self.client = server_public

            def callback(self, domain, key):
                if key == self.client:
                    return True
                else:
                    return False

            async def async_callback(self, domain, key):
                await asyncio.sleep(0.1)
                if key == self.client:
                    return True
                else:
                    return False
        if async_:
            CredentialsProvider.callback = CredentialsProvider.async_callback
        provider = CredentialsProvider()
        self.auth.configure_curve_callback(credentials_provider=provider)
        with self.curve_push_pull(certs) as (server, client):
            if key == 'ok':
                assert await self.can_connect(server, client)
            else:
                assert not await self.can_connect(server, client, timeout=200)

    @skip_pypy
    async def test_curve_user_id(self, certs, public_keys_dir):
        """threaded auth - CURVE"""
        self.auth.allow('127.0.0.1')
        server_public, server_secret, client_public, client_secret = certs
        self.auth.configure_curve(domain='*', location=public_keys_dir)
        with self.push_pull() as (client, server):
            server.curve_publickey = server_public
            server.curve_secretkey = server_secret
            server.curve_server = True
            client.curve_publickey = client_public
            client.curve_secretkey = client_secret
            client.curve_serverkey = server_public
            assert await self.can_connect(client, server)
            await client.send(b'test')
            msg = await server.recv(copy=False)
            assert msg.bytes == b'test'
            try:
                user_id = msg.get('User-Id')
            except zmq.ZMQVersionError:
                pass
            else:
                assert user_id == client_public.decode('utf8')
            self.auth.curve_user_id = lambda client_key: 'custom'
            with self.context.socket(zmq.PUSH) as client2:
                client2.curve_publickey = client_public
                client2.curve_secretkey = client_secret
                client2.curve_serverkey = server_public
                assert await self.can_connect(client2, server)
                await client2.send(b'test2')
                msg = await server.recv(copy=False)
                assert msg.bytes == b'test2'
                try:
                    user_id = msg.get('User-Id')
                except zmq.ZMQVersionError:
                    pass
                else:
                    assert user_id == 'custom'