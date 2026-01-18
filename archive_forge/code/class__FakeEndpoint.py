import testscenarios
import time
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class _FakeEndpoint(object):

    def __init__(self, target=None):
        self.target = target

    def foo(self, ctxt, **kwargs):
        pass

    @rpc.expose
    def bar(self, ctxt, **kwargs):
        pass

    def _foobar(self, ctxt, **kwargs):
        pass