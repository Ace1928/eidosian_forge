from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
class TestExtensionDescriptor(base.BaseTestCase):

    def _setup_attribute_maps(self):
        self.extended_attributes = {'resource_one': {'one': 'first'}, 'resource_two': {'two': 'second'}}
        self.extension_attrs_map = {'resource_one': {'three': 'third'}}

    def test_update_attributes_map_works(self):
        self._setup_attribute_maps()
        extension_description = InheritFromExtensionDescriptor()
        extension_description.update_attributes_map(self.extended_attributes, self.extension_attrs_map)
        self.assertEqual(self.extension_attrs_map, {'resource_one': {'one': 'first', 'three': 'third'}})

    def test_update_attributes_map_short_circuit_exit(self):
        self._setup_attribute_maps()
        extension_description = InheritFromExtensionDescriptor()
        extension_description.update_attributes_map(self.extended_attributes)
        self.assertEqual(self.extension_attrs_map, {'resource_one': {'three': 'third'}})