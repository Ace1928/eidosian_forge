import threading
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging.rpc import dispatcher
from oslo_messaging.rpc import server as rpc_server_module
from oslo_messaging import server as server_module
from oslo_messaging.tests import utils as test_utils
class TestRPCExposeDecorator(test_utils.BaseTestCase):

    def foo(self):
        pass

    @rpc.expose
    def bar(self):
        """bar docstring"""
        pass

    def test_undecorated(self):
        self.assertRaises(AttributeError, lambda: self.foo.exposed)

    def test_decorated(self):
        self.assertEqual(True, self.bar.exposed)
        self.assertEqual('bar docstring', self.bar.__doc__)
        self.assertEqual('bar', self.bar.__name__)