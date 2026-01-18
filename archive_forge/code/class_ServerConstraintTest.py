import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class ServerConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(ServerConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_get_server = mock.Mock()
        self.ctx.clients.client_plugin('nova').get_server = self.mock_get_server
        self.constraint = nova.ServerConstraint()

    def test_validation(self):
        self.mock_get_server.return_value = mock.MagicMock()
        self.assertTrue(self.constraint.validate('foo', self.ctx))

    def test_validation_error(self):
        self.mock_get_server.side_effect = exception.EntityNotFound(entity='Server', name='bar')
        self.assertFalse(self.constraint.validate('bar', self.ctx))