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
def _load_waiters(self, attrs, resource_name, resource_model, service_context):
    """
        Load resource waiters from the model. Each waiter allows you to
        wait until a resource reaches a specific state by polling the state
        of the resource.
        """
    for waiter in resource_model.waiters:
        attrs[waiter.name] = self._create_waiter(resource_waiter_model=waiter, resource_name=resource_name, service_context=service_context)