import copy
import logging
from botocore import xform_name
from botocore.utils import merge_dicts
from .action import BatchAction
from .params import create_request_parameters
from .response import ResourceHandler
from ..docs import docstring
def _load_batch_actions(self, attrs, resource_name, collection_model, service_model, event_emitter):
    """
        Batch actions on the collection become methods on both
        the collection manager and iterators.
        """
    for action_model in collection_model.batch_actions:
        snake_cased = xform_name(action_model.name)
        attrs[snake_cased] = self._create_batch_action(resource_name, snake_cased, action_model, collection_model, service_model, event_emitter)