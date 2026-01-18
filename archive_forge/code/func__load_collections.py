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
def _load_collections(self, attrs, resource_model, service_context):
    """
        Load resource collections from the model. Each collection becomes
        a :py:class:`~boto3.resources.collection.CollectionManager` instance
        on the resource instance, which allows you to iterate and filter
        through the collection's items.
        """
    for collection_model in resource_model.collections:
        attrs[collection_model.name] = self._create_collection(resource_name=resource_model.name, collection_model=collection_model, service_context=service_context)