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
def _load_attributes(self, attrs, meta, resource_name, resource_model, service_context):
    """
        Load resource attributes based on the resource shape. The shape
        name is referenced in the resource JSON, but the shape itself
        is defined in the Botocore service JSON, hence the need for
        access to the ``service_model``.
        """
    if not resource_model.shape:
        return
    shape = service_context.service_model.shape_for(resource_model.shape)
    identifiers = dict(((i.member_name, i) for i in resource_model.identifiers if i.member_name))
    attributes = resource_model.get_attributes(shape)
    for name, (orig_name, member) in attributes.items():
        if name in identifiers:
            prop = self._create_identifier_alias(resource_name=resource_name, identifier=identifiers[name], member_model=member, service_context=service_context)
        else:
            prop = self._create_autoload_property(resource_name=resource_name, name=orig_name, snake_cased=name, member_model=member, service_context=service_context)
        attrs[name] = prop