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
class ClientChannelTestCase(base.BaseTestCase):
    DICT = {'string_1': ('tuple_1', b'tuple_2'), b'byte_1': ['list_1', 'list_2']}
    EXPECTED = {'string_1': ('tuple_1', b'tuple_2'), 'byte_1': ['list_1', 'list_2']}

    def setUp(self):
        super(ClientChannelTestCase, self).setUp()
        context = get_fake_context()
        with mock.patch.object(comm.ClientChannel, '__init__'), mock.patch.object(daemon._ClientChannel, 'exchange_ping'):
            self.client_channel = daemon._ClientChannel(mock.ANY, context)

    @mock.patch.object(daemon.LOG.logger, 'handle')
    def test_out_of_band_log_message(self, handle_mock):
        message = [comm.Message.LOG, self.DICT]
        self.assertEqual(self.client_channel.log, daemon.LOG)
        with mock.patch.object(pylogging, 'makeLogRecord') as mock_make_log, mock.patch.object(daemon.LOG, 'isEnabledFor', return_value=True) as mock_enabled:
            self.client_channel.out_of_band(message)
            mock_make_log.assert_called_once_with(self.EXPECTED)
            handle_mock.assert_called_once_with(mock_make_log.return_value)
            mock_enabled.assert_called_once_with(mock_make_log.return_value.levelno)

    def test_out_of_band_not_log_message(self):
        with mock.patch.object(daemon.LOG, 'warning') as mock_warning:
            self.client_channel.out_of_band([comm.Message.PING])
            mock_warning.assert_called_once()

    @mock.patch.object(daemon.logging, 'getLogger')
    @mock.patch.object(pylogging, 'makeLogRecord')
    def test_out_of_band_log_message_context_logger(self, make_log_mock, get_logger_mock):
        logger_name = 'os_brick.privileged'
        context = get_fake_context(conf_attrs={'logger_name': logger_name})
        with mock.patch.object(comm.ClientChannel, '__init__'), mock.patch.object(daemon._ClientChannel, 'exchange_ping'):
            channel = daemon._ClientChannel(mock.ANY, context)
        get_logger_mock.assert_called_once_with(logger_name)
        self.assertEqual(get_logger_mock.return_value, channel.log)
        message = [comm.Message.LOG, self.DICT]
        channel.out_of_band(message)
        make_log_mock.assert_called_once_with(self.EXPECTED)
        channel.log.isEnabledFor.assert_called_once_with(make_log_mock.return_value.levelno)
        channel.log.logger.handle.assert_called_once_with(make_log_mock.return_value)