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
class FlavorConstraintTest(common.HeatTestCase):

    def test_validate(self):
        client = fakes_nova.FakeClient()
        self.stub_keystoneclient()
        self.patchobject(nova.NovaClientPlugin, 'get_max_microversion', return_value='2.27')
        self.patchobject(nova.NovaClientPlugin, '_create', return_value=client)
        client.flavors = mock.MagicMock()
        flavor = collections.namedtuple('Flavor', ['id', 'name'])
        flavor.id = '1234'
        flavor.name = 'foo'
        client.flavors.get.side_effect = [flavor, nova_exceptions.NotFound(''), nova_exceptions.NotFound('')]
        client.flavors.find.side_effect = [flavor, nova_exceptions.NotFound('')]
        constraint = nova.FlavorConstraint()
        ctx = utils.dummy_context()
        self.assertTrue(constraint.validate('1234', ctx))
        self.assertTrue(constraint.validate('foo', ctx))
        self.assertFalse(constraint.validate('bar', ctx))
        self.assertEqual(1, nova.NovaClientPlugin._create.call_count)
        self.assertEqual(3, client.flavors.get.call_count)
        self.assertEqual(2, client.flavors.find.call_count)