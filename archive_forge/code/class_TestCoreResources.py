from oslo_utils import uuidutils
import testtools
from webob import exc
from neutron_lib.api import attributes
from neutron_lib.api import converters
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib import constants
from neutron_lib import context
from neutron_lib import exceptions
from neutron_lib.tests import _base as base
class TestCoreResources(base.BaseTestCase):
    CORE_DEFS = [network, port, subnet, subnetpool]

    def test_core_resource_names(self):
        self.assertEqual(sorted([r.COLLECTION_NAME for r in TestCoreResources.CORE_DEFS]), sorted(attributes.RESOURCES.keys()))

    def test_core_resource_attrs(self):
        for r in TestCoreResources.CORE_DEFS:
            self.assertIs(r.RESOURCE_ATTRIBUTE_MAP[r.COLLECTION_NAME], attributes.RESOURCES[r.COLLECTION_NAME])