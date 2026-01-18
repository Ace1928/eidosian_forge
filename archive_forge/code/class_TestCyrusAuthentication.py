import copy
import logging
import os
import queue
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
import uuid
from oslo_utils import eventletutils
from oslo_utils import importutils
from string import Template
import testtools
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
@testtools.skipUnless(CYRUS_ENABLED, 'Cyrus SASL not supported')
class TestCyrusAuthentication(test_utils.BaseTestCase):
    """Test the driver's Cyrus SASL integration"""
    _conf_dir = None
    _mechs = 'DIGEST-MD5 SCRAM-SHA-1 CRAM-MD5 PLAIN'

    @classmethod
    def setUpClass(cls):
        cls._conf_dir = '/tmp/amqp1_tests_%s' % os.getpid()
        os.makedirs(cls._conf_dir)
        db = os.path.join(cls._conf_dir, 'openstack.sasldb')
        _t = 'echo secret | saslpasswd2 -c -p -f ${db} -u myrealm joe'
        cmd = Template(_t).substitute(db=db)
        try:
            subprocess.check_call(args=cmd, shell=True)
        except Exception:
            shutil.rmtree(cls._conf_dir, ignore_errors=True)
            cls._conf_dir = None
            return
        conf = os.path.join(cls._conf_dir, 'openstack.conf')
        t = Template('sasldb_path: ${db}\npwcheck_method: auxprop\nauxprop_plugin: sasldb\nmech_list: ${mechs}\n')
        with open(conf, 'w') as f:
            f.write(t.substitute(db=db, mechs=cls._mechs))

    @classmethod
    def tearDownClass(cls):
        if cls._conf_dir:
            shutil.rmtree(cls._conf_dir, ignore_errors=True)

    def setUp(self):
        super(TestCyrusAuthentication, self).setUp()
        if TestCyrusAuthentication._conf_dir is None:
            self.skipTest('Cyrus SASL tools not installed')
        _mechs = TestCyrusAuthentication._mechs
        _dir = TestCyrusAuthentication._conf_dir
        self._broker = FakeBroker(self.conf.oslo_messaging_amqp, sasl_mechanisms=_mechs, user_credentials=['\x00joe@myrealm\x00secret'], sasl_config_dir=_dir, sasl_config_name='openstack')
        self._broker.start()
        self.messaging_conf.transport_url = 'amqp://'
        self.conf = self.messaging_conf.conf

    def tearDown(self):
        if self._broker:
            self._broker.stop()
            self._broker = None
        super(TestCyrusAuthentication, self).tearDown()

    def _authentication_test(self, addr, retry=None):
        url = oslo_messaging.TransportURL.parse(self.conf, addr)
        driver = amqp_driver.ProtonDriver(self.conf, url)
        target = oslo_messaging.Target(topic='test-topic')
        listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1)
        try:
            rc = driver.send(target, {'context': True}, {'method': 'echo'}, wait_for_reply=True, retry=retry)
            self.assertIsNotNone(rc)
            listener.join(timeout=30)
            self.assertFalse(listener.is_alive())
        finally:
            driver.cleanup()

    def test_authentication_ok(self):
        """Verify that username and password given in TransportHost are
        accepted by the broker.
        """
        addr = 'amqp://joe@myrealm:secret@%s:%d' % (self._broker.host, self._broker.port)
        self._authentication_test(addr)

    def test_authentication_failure(self):
        """Verify that a bad password given in TransportHost is
        rejected by the broker.
        """
        addr = 'amqp://joe@myrealm:badpass@%s:%d' % (self._broker.host, self._broker.port)
        try:
            self._authentication_test(addr, retry=2)
        except oslo_messaging.MessageDeliveryFailure as e:
            self.assertTrue('amqp:unauthorized-access' in str(e))
        else:
            self.assertIsNone('Expected authentication failure')

    def test_authentication_bad_mechs(self):
        """Verify that the connection fails if the client's SASL mechanisms do
        not match the broker's.
        """
        self.config(sasl_mechanisms='EXTERNAL ANONYMOUS', group='oslo_messaging_amqp')
        addr = 'amqp://joe@myrealm:secret@%s:%d' % (self._broker.host, self._broker.port)
        self.assertRaises(oslo_messaging.MessageDeliveryFailure, self._authentication_test, addr, retry=0)

    def test_authentication_default_realm(self):
        """Verify that default realm is used if none present in username"""
        addr = 'amqp://joe:secret@%s:%d' % (self._broker.host, self._broker.port)
        self.config(sasl_default_realm='myrealm', group='oslo_messaging_amqp')
        self._authentication_test(addr)

    def test_authentication_ignore_default_realm(self):
        """Verify that default realm is not used if realm present in
        username
        """
        addr = 'amqp://joe@myrealm:secret@%s:%d' % (self._broker.host, self._broker.port)
        self.config(sasl_default_realm='bad-realm', group='oslo_messaging_amqp')
        self._authentication_test(addr)