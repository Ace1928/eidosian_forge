import json
import logging
import types
from ray import cloudpickle as cloudpickle
from ray._private.utils import binary_to_hex, hex_to_binary
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once
def _to_cloudpickle(self, obj):
    return {'_type': 'CLOUDPICKLE_FALLBACK', 'value': binary_to_hex(cloudpickle.dumps(obj))}