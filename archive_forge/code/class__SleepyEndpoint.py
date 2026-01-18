import testscenarios
import time
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class _SleepyEndpoint(object):

    def __init__(self, target=None):
        self.target = target

    def sleep(self, ctxt, **kwargs):
        time.sleep(kwargs['timeout'])
        return True