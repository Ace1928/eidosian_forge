import testscenarios
import time
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def _set_endpoint_mock_properties(endpoint):
    endpoint.foo = mock.Mock(spec=dir(_FakeEndpoint.foo))
    endpoint.bar = mock.Mock(spec=dir(_FakeEndpoint.bar))
    endpoint.bar.exposed = mock.PropertyMock(return_value=True)
    endpoint._foobar = mock.Mock(spec=dir(_FakeEndpoint._foobar))
    return endpoint