import json
import logging
import types
from ray import cloudpickle as cloudpickle
from ray._private.utils import binary_to_hex, hex_to_binary
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once
@DeveloperAPI
class TuneFunctionDecoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, *args, object_hook=self.object_hook, **kwargs)

    def object_hook(self, obj):
        if obj.get('_type') == 'CLOUDPICKLE_FALLBACK':
            return self._from_cloudpickle(obj)
        return obj

    def _from_cloudpickle(self, obj):
        return cloudpickle.loads(hex_to_binary(obj['value']))