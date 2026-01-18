import logging
from functools import partial
from .action import ServiceAction
from .action import WaiterAction
from .base import ResourceMeta, ServiceResource
from .collection import CollectionFactory
from .model import ResourceModel
from .response import build_identifiers, ResourceHandler
from ..exceptions import ResourceLoadException
from ..docs import docstring
def _create_reference(factory_self, reference_model, resource_name, service_context):
    """
        Creates a new property on the resource to lazy-load a reference.
        """
    handler = ResourceHandler(search_path=reference_model.resource.path, factory=factory_self, resource_model=reference_model.resource, service_context=service_context)
    needs_data = any((i.source == 'data' for i in reference_model.resource.identifiers))

    def get_reference(self):
        if needs_data and self.meta.data is None and hasattr(self, 'load'):
            self.load()
        return handler(self, {}, self.meta.data)
    get_reference.__name__ = str(reference_model.name)
    get_reference.__doc__ = docstring.ReferenceDocstring(reference_model=reference_model, include_signature=False)
    return property(get_reference)