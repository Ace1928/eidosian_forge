import datetime
import logging
import sys
import uuid
import fixtures
from kombu import connection
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import timeutils
from stevedore import dispatch
from stevedore import extension
import testscenarios
import yaml
import oslo_messaging
from oslo_messaging.notify import _impl_log
from oslo_messaging.notify import _impl_test
from oslo_messaging.notify import messaging
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class TestNotifierTransportWarning(test_utils.BaseTestCase):

    @mock.patch('oslo_messaging.notify.notifier._LOG')
    def test_warning_when_rpc_transport(self, log):
        transport = oslo_messaging.get_rpc_transport(self.conf)
        oslo_messaging.Notifier(transport, 'test.localhost')
        log.warning.assert_called_once_with('Using RPC transport for notifications. Please use get_notification_transport to obtain a notification transport instance.')