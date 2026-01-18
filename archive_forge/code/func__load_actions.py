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
def _load_actions(self, attrs, resource_name, resource_model, service_context):
    """
        Actions on the resource become methods, with the ``load`` method
        being a special case which sets internal data for attributes, and
        ``reload`` is an alias for ``load``.
        """
    if resource_model.load:
        attrs['load'] = self._create_action(action_model=resource_model.load, resource_name=resource_name, service_context=service_context, is_load=True)
        attrs['reload'] = attrs['load']
    for action in resource_model.actions:
        attrs[action.name] = self._create_action(action_model=action, resource_name=resource_name, service_context=service_context)