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
def _load_has_relations(self, attrs, resource_name, resource_model, service_context):
    """
        Load related resources, which are defined via a ``has``
        relationship but conceptually come in two forms:

        1. A reference, which is a related resource instance and can be
           ``None``, such as an EC2 instance's ``vpc``.
        2. A subresource, which is a resource constructor that will always
           return a resource instance which shares identifiers/data with
           this resource, such as ``s3.Bucket('name').Object('key')``.
        """
    for reference in resource_model.references:
        attrs[reference.name] = self._create_reference(reference_model=reference, resource_name=resource_name, service_context=service_context)
    for subresource in resource_model.subresources:
        attrs[subresource.name] = self._create_class_partial(subresource_model=subresource, resource_name=resource_name, service_context=service_context)
    self._create_available_subresources_command(attrs, resource_model.subresources)