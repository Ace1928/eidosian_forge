import json
import logging
import types
from ray import cloudpickle as cloudpickle
from ray._private.utils import binary_to_hex, hex_to_binary
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once
@DeveloperAPI
class TuneFunctionEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, types.FunctionType):
            return self._to_cloudpickle(obj)
        try:
            return super(TuneFunctionEncoder, self).default(obj)
        except Exception:
            if log_once(f'tune_func_encode:{str(obj)}'):
                logger.debug('Unable to encode. Falling back to cloudpickle.')
            return self._to_cloudpickle(obj)

    def _to_cloudpickle(self, obj):
        return {'_type': 'CLOUDPICKLE_FALLBACK', 'value': binary_to_hex(cloudpickle.dumps(obj))}