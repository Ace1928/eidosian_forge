from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
def _get_combined_properties(self, dev):
    """Combine the current DeviceSpec with another DeviceSpec.

    The combination of DeviceSpecs is will give priority to dev.

    Args:
      dev: a `DeviceSpec`

    Returns:
      A tuple of (job, replica, task, device_type, device_index) which
      represents the combination of self and dev.
    """
    return (dev.job if dev.job is not None else self.job, dev.replica if dev.replica is not None else self.replica, dev.task if dev.task is not None else self.task, dev.device_type if dev.device_type is not None else self.device_type, dev.device_index if dev.device_index is not None else self.device_index)