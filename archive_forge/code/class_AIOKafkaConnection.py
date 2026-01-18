import asyncio
import collections
import base64
import functools
import hashlib
import hmac
import logging
import random
import socket
import struct
import sys
import time
import traceback
import uuid
import warnings
import weakref
import async_timeout
import aiokafka.errors as Errors
from aiokafka.abc import AbstractTokenProvider
from aiokafka.protocol.api import RequestHeader
from aiokafka.protocol.admin import (
from aiokafka.protocol.commit import (
from aiokafka.util import create_future, create_task, get_running_loop, wait_for
class AIOKafkaConnection:
    """Class for manage connection to Kafka node"""
    _reader = None
    _source_traceback = None

    def __init__(self, host, port, *, client_id='aiokafka', request_timeout_ms=40000, api_version=(0, 8, 2), ssl_context=None, security_protocol='PLAINTEXT', max_idle_ms=None, on_close=None, sasl_mechanism=None, sasl_plain_password=None, sasl_plain_username=None, sasl_kerberos_service_name='kafka', sasl_kerberos_domain_name=None, sasl_oauth_token_provider=None, version_hint=None):
        loop = get_running_loop()
        if sasl_mechanism == 'GSSAPI':
            assert gssapi is not None, 'gssapi library required'
        if sasl_mechanism == 'OAUTHBEARER':
            if sasl_oauth_token_provider is None or not isinstance(sasl_oauth_token_provider, AbstractTokenProvider):
                raise ValueError('sasl_oauth_token_provider needs to be                     provided implementing aiokafka.abc.AbstractTokenProvider')
            assert callable(getattr(sasl_oauth_token_provider, 'token', None)), 'sasl_oauth_token_provider must implement method #token()'
        self._loop = loop
        self._host = host
        self._port = port
        self._request_timeout = request_timeout_ms / 1000
        self._api_version = api_version
        self._client_id = client_id
        self._ssl_context = ssl_context
        self._security_protocol = security_protocol
        self._sasl_mechanism = sasl_mechanism
        self._sasl_plain_username = sasl_plain_username
        self._sasl_plain_password = sasl_plain_password
        self._sasl_kerberos_service_name = sasl_kerberos_service_name
        self._sasl_kerberos_domain_name = sasl_kerberos_domain_name
        self._sasl_oauth_token_provider = sasl_oauth_token_provider
        self._version_hint = version_hint
        self._version_info = VersionInfo({})
        self._reader = self._writer = self._protocol = None
        self._requests = collections.deque()
        self._read_task = None
        self._correlation_id = 0
        self._closed_fut = None
        self._max_idle_ms = max_idle_ms
        self._last_action = time.monotonic()
        self._idle_handle = None
        self._on_close_cb = on_close
        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))

    def __del__(self, _warnings=warnings):
        if self.connected():
            _warnings.warn(f'Unclosed AIOKafkaConnection {self!r}', ResourceWarning, source=self)
            if self._loop.is_closed():
                return
            self._on_close_cb = None
            self.close()
            context = {'conn': self, 'message': 'Unclosed AIOKafkaConnection'}
            if self._source_traceback is not None:
                context['source_traceback'] = self._source_traceback
            self._loop.call_exception_handler(context)

    async def connect(self):
        loop = self._loop
        self._closed_fut = create_future()
        if self._security_protocol in ['PLAINTEXT', 'SASL_PLAINTEXT']:
            ssl = None
        else:
            assert self._security_protocol in ['SSL', 'SASL_SSL']
            assert self._ssl_context is not None
            ssl = self._ssl_context
        reader = asyncio.StreamReader(limit=READER_LIMIT, loop=loop)
        protocol = AIOKafkaProtocol(self._closed_fut, reader, loop=loop)
        async with async_timeout.timeout(self._request_timeout):
            transport, _ = await loop.create_connection(lambda: protocol, self.host, self.port, ssl=ssl)
        writer = asyncio.StreamWriter(transport, protocol, reader, loop)
        self._reader, self._writer, self._protocol = (reader, writer, protocol)
        self._read_task = self._create_reader_task()
        if self._max_idle_ms is not None:
            self._idle_handle = loop.call_soon(self._idle_check, weakref.ref(self))
        try:
            if self._version_hint and self._version_hint >= (0, 10):
                await self._do_version_lookup()
            if self._security_protocol in ['SASL_SSL', 'SASL_PLAINTEXT']:
                await self._do_sasl_handshake()
        except:
            self.close()
            raise
        return (reader, writer)

    async def _do_version_lookup(self):
        version_req = ApiVersionRequest[0]()
        response = await self.send(version_req)
        versions = {}
        for api_key, min_version, max_version in response.api_versions:
            assert min_version <= max_version, f'{min_version} should be less than or equal to {max_version} for {api_key}'
            versions[api_key] = (min_version, max_version)
        self._version_info = VersionInfo(versions)

    async def _do_sasl_handshake(self):
        if self._version_hint and self._version_hint < (0, 10):
            handshake_klass = None
            assert self._sasl_mechanism == 'GSSAPI', 'Only GSSAPI supported for v0.9'
        else:
            handshake_klass = self._version_info.pick_best(SaslHandShakeRequest)
            sasl_handshake = handshake_klass(self._sasl_mechanism)
            response = await self.send(sasl_handshake)
            error_type = Errors.for_code(response.error_code)
            if error_type is not Errors.NoError:
                error = error_type(self)
                self.close(reason=CloseReason.AUTH_FAILURE, exc=error)
                raise error
            if self._sasl_mechanism not in response.enabled_mechanisms:
                exc = Errors.UnsupportedSaslMechanismError(f'Kafka broker does not support {self._sasl_mechanism} sasl mechanism. Enabled mechanisms are: {response.enabled_mechanisms}')
                self.close(reason=CloseReason.AUTH_FAILURE, exc=exc)
                raise exc
        assert self._sasl_mechanism in ('PLAIN', 'GSSAPI', 'SCRAM-SHA-256', 'SCRAM-SHA-512', 'OAUTHBEARER')
        if self._security_protocol == 'SASL_PLAINTEXT' and self._sasl_mechanism == 'PLAIN':
            log.warning('Sending username and password in the clear')
        if self._sasl_mechanism == 'GSSAPI':
            authenticator = self.authenticator_gssapi()
        elif self._sasl_mechanism.startswith('SCRAM-SHA-'):
            authenticator = self.authenticator_scram()
        elif self._sasl_mechanism == 'OAUTHBEARER':
            authenticator = self.authenticator_oauth()
        else:
            authenticator = self.authenticator_plain()
        if handshake_klass is not None and sasl_handshake.API_VERSION > 0:
            auth_klass = self._version_info.pick_best(SaslAuthenticateRequest)
        else:
            auth_klass = None
        auth_bytes = None
        expect_response = True
        while True:
            res = await authenticator.step(auth_bytes)
            if res is None:
                break
            payload, expect_response = res
            if auth_klass is None:
                auth_bytes = await self._send_sasl_token(payload, expect_response)
            else:
                req = auth_klass(payload)
                resp = await self.send(req)
                error_type = Errors.for_code(resp.error_code)
                if error_type is not Errors.NoError:
                    exc = error_type(resp.error_message)
                    self.close(reason=CloseReason.AUTH_FAILURE, exc=exc)
                    raise exc
                auth_bytes = resp.sasl_auth_bytes
        if self._sasl_mechanism == 'GSSAPI':
            log.info('Authenticated as %s via GSSAPI', self.sasl_principal)
        elif self._sasl_mechanism == 'OAUTHBEARER':
            log.info('Authenticated via OAUTHBEARER')
        else:
            log.info('Authenticated as %s via %s', self._sasl_plain_username, self._sasl_mechanism)

    def authenticator_plain(self):
        return SaslPlainAuthenticator(loop=self._loop, sasl_plain_password=self._sasl_plain_password, sasl_plain_username=self._sasl_plain_username)

    def authenticator_gssapi(self):
        return SaslGSSAPIAuthenticator(loop=self._loop, principal=self.sasl_principal)

    def authenticator_scram(self):
        return ScramAuthenticator(loop=self._loop, sasl_plain_password=self._sasl_plain_password, sasl_plain_username=self._sasl_plain_username, sasl_mechanism=self._sasl_mechanism)

    def authenticator_oauth(self):
        return OAuthAuthenticator(sasl_oauth_token_provider=self._sasl_oauth_token_provider)

    @property
    def sasl_principal(self):
        service = self._sasl_kerberos_service_name
        domain = self._sasl_kerberos_domain_name or self.host
        return f'{service}@{domain}'

    @classmethod
    def _on_read_task_error(cls, self_ref, read_task):
        if read_task.cancelled():
            return
        try:
            read_task.result()
        except Exception as exc:
            if not isinstance(exc, (OSError, EOFError, ConnectionError)):
                log.exception('Unexpected exception in AIOKafkaConnection')
            self = self_ref()
            if self is not None:
                self.close(reason=CloseReason.CONNECTION_BROKEN, exc=exc)

    @staticmethod
    def _idle_check(self_ref):
        self = self_ref()
        if self is None:
            return
        idle_for = time.monotonic() - self._last_action
        timeout = self._max_idle_ms / 1000
        if idle_for >= timeout and (not self._requests):
            self.close(CloseReason.IDLE_DROP)
        else:
            if self._requests:
                wake_up_in = timeout
            else:
                wake_up_in = timeout - idle_for
            self._idle_handle = self._loop.call_later(wake_up_in, self._idle_check, self_ref)

    def __repr__(self):
        return f'<AIOKafkaConnection host={self.host} port={self.port}>'

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    def send(self, request, expect_response=True):
        if self._writer is None:
            raise Errors.KafkaConnectionError(f'No connection to broker at {self._host}:{self._port}')
        correlation_id = self._next_correlation_id()
        header = RequestHeader(request, correlation_id=correlation_id, client_id=self._client_id)
        message = header.encode() + request.encode()
        size = struct.pack('>i', len(message))
        try:
            self._writer.write(size + message)
        except OSError as err:
            self.close(reason=CloseReason.CONNECTION_BROKEN)
            raise Errors.KafkaConnectionError(f'Connection at {self._host}:{self._port} broken: {err}')
        log.debug('%s Request %d: %s', self, correlation_id, request)
        if not expect_response:
            return self._writer.drain()
        fut = self._loop.create_future()
        self._requests.append((correlation_id, request.RESPONSE_TYPE, fut))
        return wait_for(fut, self._request_timeout)

    def _send_sasl_token(self, payload, expect_response=True):
        if self._writer is None:
            raise Errors.KafkaConnectionError(f'No connection to broker at {self._host}:{self._port}')
        size = struct.pack('>i', len(payload))
        try:
            self._writer.write(size + payload)
        except OSError as err:
            self.close(reason=CloseReason.CONNECTION_BROKEN)
            raise Errors.KafkaConnectionError(f'Connection at {self._host}:{self._port} broken: {err}')
        if not expect_response:
            return self._writer.drain()
        fut = self._loop.create_future()
        self._requests.append((None, None, fut))
        return wait_for(fut, self._request_timeout)

    def connected(self):
        return bool(self._reader is not None and (not self._reader.at_eof()))

    def close(self, reason=None, exc=None):
        log.debug('Closing connection at %s:%s', self._host, self._port)
        if self._reader is not None:
            self._writer.close()
            self._writer = self._reader = None
            if not self._read_task.done():
                self._read_task.cancel()
                self._read_task = None
            for _, _, fut in self._requests:
                if not fut.done():
                    error = Errors.KafkaConnectionError(f'Connection at {self._host}:{self._port} closed')
                    if exc is not None:
                        error.__cause__ = exc
                        error.__context__ = exc
                    fut.set_exception(error)
            self._requests = collections.deque()
            if self._on_close_cb is not None:
                self._on_close_cb(self, reason)
                self._on_close_cb = None
        if self._idle_handle is not None:
            self._idle_handle.cancel()
        return self._closed_fut

    def _create_reader_task(self):
        self_ref = weakref.ref(self)
        read_task = create_task(self._read(self_ref))
        read_task.add_done_callback(functools.partial(self._on_read_task_error, self_ref))
        return read_task

    @staticmethod
    async def _read(self_ref):
        self = self_ref()
        if self is None:
            return
        reader = self._reader
        del self
        while True:
            resp = await reader.readexactly(4)
            size, = struct.unpack('>i', resp)
            resp = await reader.readexactly(size)
            self = self_ref()
            if self is None:
                return
            self._handle_frame(resp)
            del self

    def _handle_frame(self, resp):
        correlation_id, resp_type, fut = self._requests[0]
        if correlation_id is None:
            if not fut.done():
                fut.set_result(resp)
        else:
            recv_correlation_id, = struct.unpack_from('>i', resp, 0)
            if self._api_version == (0, 8, 2) and resp_type is GroupCoordinatorResponse and (correlation_id != 0) and (recv_correlation_id == 0):
                log.warning('Kafka 0.8.2 quirk -- GroupCoordinatorResponse coorelation id does not match request. This should go away once at least one topic has been initialized on the broker')
            elif correlation_id != recv_correlation_id:
                error = Errors.CorrelationIdError(f'Correlation ids do not match: sent {correlation_id}, recv {recv_correlation_id}')
                if not fut.done():
                    fut.set_exception(error)
                self.close(reason=CloseReason.OUT_OF_SYNC)
                return
            if not fut.done():
                response = resp_type.decode(resp[4:])
                log.debug('%s Response %d: %s', self, correlation_id, response)
                fut.set_result(response)
        self._last_action = time.monotonic()
        self._requests.popleft()

    def _next_correlation_id(self):
        self._correlation_id = (self._correlation_id + 1) % 2 ** 31
        return self._correlation_id