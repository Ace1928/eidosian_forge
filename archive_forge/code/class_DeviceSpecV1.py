from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
@tf_export(v1=['DeviceSpec'])
class DeviceSpecV1(DeviceSpecV2):
    __doc__ = DeviceSpecV2.__doc__
    __slots__ = DeviceSpecV2.__slots__

    @DeviceSpecV2.job.setter
    def job(self, job):
        self._job = _as_str_or_none(job)
        self._as_string, self._hash = (None, None)

    @DeviceSpecV2.replica.setter
    def replica(self, replica):
        self._replica = _as_int_or_none(replica)
        self._as_string, self._hash = (None, None)

    @DeviceSpecV2.task.setter
    def task(self, task):
        self._task = _as_int_or_none(task)
        self._as_string, self._hash = (None, None)

    @DeviceSpecV2.device_type.setter
    def device_type(self, device_type):
        self._device_type = _as_device_str_or_none(device_type)
        self._as_string, self._hash = (None, None)

    @DeviceSpecV2.device_index.setter
    def device_index(self, device_index):
        self._device_index = _as_int_or_none(device_index)
        self._as_string, self._hash = (None, None)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.to_string())
        return self._hash

    def to_string(self):
        if self._as_string is None:
            self._as_string = self._components_to_string(job=self.job, replica=self.replica, task=self.task, device_type=self.device_type, device_index=self.device_index)
        return self._as_string

    def parse_from_string(self, spec):
        self.job, self.replica, self.task, self.device_type, self.device_index = self._string_to_components(spec)
        return self

    def merge_from(self, dev):
        """Merge the properties of "dev" into this `DeviceSpec`.

    Note: Will be removed in TensorFlow 2.x since DeviceSpecs will become
          immutable.

    Args:
      dev: a `DeviceSpec`.
    """
        self.job, self.replica, self.task, self.device_type, self.device_index = self._get_combined_properties(dev)
    to_string.__doc__ = DeviceSpecV2.to_string.__doc__
    parse_from_string.__doc__ = DeviceSpecV2.parse_from_string.__doc__