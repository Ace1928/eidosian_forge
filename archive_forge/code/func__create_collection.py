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
def _create_collection(factory_self, resource_name, collection_model, service_context):
    """
        Creates a new property on the resource to lazy-load a collection.
        """
    cls = factory_self._collection_factory.load_from_definition(resource_name=resource_name, collection_model=collection_model, service_context=service_context, event_emitter=factory_self._emitter)

    def get_collection(self):
        return cls(collection_model=collection_model, parent=self, factory=factory_self, service_context=service_context)
    get_collection.__name__ = str(collection_model.name)
    get_collection.__doc__ = docstring.CollectionDocstring(collection_model=collection_model, include_signature=False)
    return property(get_collection)