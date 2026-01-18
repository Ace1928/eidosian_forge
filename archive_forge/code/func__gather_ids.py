import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
def _gather_ids(self, shape, params, ids=None):
    if ids is None:
        ids = {}
    for member_name, member_shape in shape.members.items():
        if member_shape.metadata.get('endpointdiscoveryid'):
            ids[member_name] = params[member_name]
        elif member_shape.type_name == 'structure' and member_name in params:
            self._gather_ids(member_shape, params[member_name], ids)
    return ids