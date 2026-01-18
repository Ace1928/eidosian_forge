from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
class InheritFromExtensionDescriptor(extensions.ExtensionDescriptor):
    """Class to inherit from ExtensionDescriptor to test its methods

    Because ExtensionDescriptor is an abstract class, in order to test methods
    we need to create something to inherit from it so we have something
    instantiatable.  The only things defined here are those that are required
    because in ExtensionDescriptor they are marked as @abc.abstractmethod.
    """

    def get_name(self):
        pass

    def get_alias(self):
        pass

    def get_description(self):
        pass

    def get_updated(self):
        pass

    def update_attributes_map_save(self, extended_attributes, extension_attrs_map=None):
        """Update attributes map for this extension.

        This is default method for extending an extension's attributes map.
        An extension can use this method and supplying its own resource
        attribute map in extension_attrs_map argument to extend all its
        attributes that needs to be extended.

        If an extension does not implement update_attributes_map, the method
        does nothing and just return.
        """
        if not extension_attrs_map:
            return
        for resource, attrs in extension_attrs_map.items():
            extended_attrs = extended_attributes.get(resource)
            if extended_attrs:
                attrs.update(extended_attrs)