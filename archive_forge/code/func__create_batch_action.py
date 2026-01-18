import copy
import logging
from botocore import xform_name
from botocore.utils import merge_dicts
from .action import BatchAction
from .params import create_request_parameters
from .response import ResourceHandler
from ..docs import docstring
def _create_batch_action(factory_self, resource_name, snake_cased, action_model, collection_model, service_model, event_emitter):
    """
        Creates a new method which makes a batch operation request
        to the underlying service API.
        """
    action = BatchAction(action_model)

    def batch_action(self, *args, **kwargs):
        return action(self, *args, **kwargs)
    batch_action.__name__ = str(snake_cased)
    batch_action.__doc__ = docstring.BatchActionDocstring(resource_name=resource_name, event_emitter=event_emitter, batch_action_model=action_model, service_model=service_model, collection_model=collection_model, include_signature=False)
    return batch_action