import logging
import os
import pipes
import platform
import sys
import tempfile
import time
from unittest import mock
import testtools
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep import priv_context
from oslo_privsep.tests import testctx
@testtools.skipIf(platform.system() != 'Linux', 'works only on Linux platform.')
class PrivContextTest(testctx.TestContextTestCase):

    @mock.patch.object(priv_context, 'sys')
    def test_init_windows(self, mock_sys):
        mock_sys.platform = 'win32'
        context = priv_context.PrivContext('test', capabilities=[])
        self.assertFalse(context.client_mode)

    @mock.patch.object(priv_context, 'sys')
    def test_set_client_mode(self, mock_sys):
        context = priv_context.PrivContext('test', capabilities=[])
        self.assertTrue(context.client_mode)
        context.set_client_mode(False)
        self.assertFalse(context.client_mode)
        mock_sys.platform = 'win32'
        self.assertRaises(RuntimeError, context.set_client_mode, True)

    def test_helper_command(self):
        self.privsep_conf.privsep.helper_command = 'foo --bar'
        _, temp_path = tempfile.mkstemp()
        cmd = testctx.context.helper_command(temp_path)
        expected = ['foo', '--bar', '--privsep_context', testctx.context.pypath, '--privsep_sock_path', temp_path]
        self.assertEqual(expected, cmd)

    def test_helper_command_default(self):
        self.privsep_conf.config_file = ['/bar.conf']
        _, temp_path = tempfile.mkstemp()
        cmd = testctx.context.helper_command(temp_path)
        expected = ['sudo', 'privsep-helper', '--config-file', '/bar.conf', '--privsep_context', testctx.context.pypath, '--privsep_sock_path', temp_path]
        self.assertEqual(expected, cmd)

    def test_helper_command_default_dirtoo(self):
        self.privsep_conf.config_file = ['/bar.conf', '/baz.conf']
        self.privsep_conf.config_dir = ['/foo.d']
        _, temp_path = tempfile.mkstemp()
        cmd = testctx.context.helper_command(temp_path)
        expected = ['sudo', 'privsep-helper', '--config-file', '/bar.conf', '--config-file', '/baz.conf', '--config-dir', '/foo.d', '--privsep_context', testctx.context.pypath, '--privsep_sock_path', temp_path]
        self.assertEqual(expected, cmd)

    def test_init_known_contexts(self):
        self.assertEqual(testctx.context.helper_command('/sock')[:2], ['sudo', 'privsep-helper'])
        priv_context.init(root_helper=['sudo', 'rootwrap'])
        self.assertEqual(testctx.context.helper_command('/sock')[:3], ['sudo', 'rootwrap', 'privsep-helper'])

    def test_start_acquires_lock(self):
        context = priv_context.PrivContext('test', capabilities=[])
        context.channel = 'something not None'
        context.start_lock = mock.Mock()
        context.start_lock.__enter__ = mock.Mock()
        context.start_lock.__exit__ = mock.Mock()
        self.assertFalse(context.start_lock.__enter__.called)
        context.start()
        self.assertTrue(context.start_lock.__enter__.called)