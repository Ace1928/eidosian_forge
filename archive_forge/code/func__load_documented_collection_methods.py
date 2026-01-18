import copy
import logging
from botocore import xform_name
from botocore.utils import merge_dicts
from .action import BatchAction
from .params import create_request_parameters
from .response import ResourceHandler
from ..docs import docstring
def _load_documented_collection_methods(factory_self, attrs, resource_name, collection_model, service_model, event_emitter, base_class):

    def all(self):
        return base_class.all(self)
    all.__doc__ = docstring.CollectionMethodDocstring(resource_name=resource_name, action_name='all', event_emitter=event_emitter, collection_model=collection_model, service_model=service_model, include_signature=False)
    attrs['all'] = all

    def filter(self, **kwargs):
        return base_class.filter(self, **kwargs)
    filter.__doc__ = docstring.CollectionMethodDocstring(resource_name=resource_name, action_name='filter', event_emitter=event_emitter, collection_model=collection_model, service_model=service_model, include_signature=False)
    attrs['filter'] = filter

    def limit(self, count):
        return base_class.limit(self, count)
    limit.__doc__ = docstring.CollectionMethodDocstring(resource_name=resource_name, action_name='limit', event_emitter=event_emitter, collection_model=collection_model, service_model=service_model, include_signature=False)
    attrs['limit'] = limit

    def page_size(self, count):
        return base_class.page_size(self, count)
    page_size.__doc__ = docstring.CollectionMethodDocstring(resource_name=resource_name, action_name='page_size', event_emitter=event_emitter, collection_model=collection_model, service_model=service_model, include_signature=False)
    attrs['page_size'] = page_size