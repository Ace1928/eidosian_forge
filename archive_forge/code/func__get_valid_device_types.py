from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
@staticmethod
def _get_valid_device_types():
    valid_device_types = set({})
    physical_devices = pywrap_tfe.TF_ListPluggablePhysicalDevices()
    for device in physical_devices:
        valid_device_types.add(device.decode().split(':')[1])
    valid_device_types = valid_device_types | _VALID_DEVICE_TYPES
    return valid_device_types