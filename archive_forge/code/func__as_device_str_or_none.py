from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
def _as_device_str_or_none(device_type):
    if device_type in ('cpu', 'gpu'):
        return device_type.upper()
    return _as_str_or_none(device_type)