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
def _setup_client(self, transport, topic='testtopic', exchange=None):
    target = oslo_messaging.Target(topic=topic, exchange=exchange)
    return oslo_messaging.get_rpc_client(transport, target=target, serializer=self.serializer)