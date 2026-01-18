import copy
import eventlet
import fixtures
import functools
import logging as pylogging
import platform
import sys
import time
from unittest import mock
from oslo_log import formatters
from oslo_log import log as logging
from oslotest import base
import testtools
from oslo_privsep import capabilities
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep.tests import testctx
@testtools.skipIf(platform.system() != 'Linux', 'works only on Linux platform.')
class DaemonTest(base.BaseTestCase):

    @mock.patch('os.setuid')
    @mock.patch('os.setgid')
    @mock.patch('os.setgroups')
    @mock.patch('oslo_privsep.capabilities.set_keepcaps')
    @mock.patch('oslo_privsep.capabilities.drop_all_caps_except')
    def test_drop_privs(self, mock_dropcaps, mock_keepcaps, mock_setgroups, mock_setgid, mock_setuid):
        channel = mock.NonCallableMock()
        context = get_fake_context()
        manager = mock.Mock()
        manager.attach_mock(mock_setuid, 'setuid')
        manager.attach_mock(mock_setgid, 'setgid')
        expected_calls = [mock.call.setgid(84), mock.call.setuid(42)]
        d = daemon.Daemon(channel, context)
        d._drop_privs()
        mock_setuid.assert_called_once_with(42)
        mock_setgid.assert_called_once_with(84)
        mock_setgroups.assert_called_once_with([])
        assert manager.mock_calls == expected_calls
        self.assertCountEqual([mock.call(True), mock.call(False)], mock_keepcaps.mock_calls)
        mock_dropcaps.assert_called_once_with(set((capabilities.CAP_SYS_ADMIN, capabilities.CAP_NET_ADMIN)), set((capabilities.CAP_SYS_ADMIN, capabilities.CAP_NET_ADMIN)), [])