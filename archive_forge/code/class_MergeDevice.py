from tensorflow.python import tf2
from tensorflow.python.framework import device_spec
class MergeDevice(object):
    """Wraps a device specification (DeviceSpec or str) with merge functionality.

  When called, this class will merge a node_def with its own spec. It also
  exposes a `shortcut_string_merge` method which can significantly improve
  performance of device placement.
  """
    __slots__ = ['_spec']

    def __init__(self, spec):
        if isinstance(spec, device_spec.DeviceSpecV2):
            self._spec = spec
        elif isinstance(spec, device_spec.DeviceSpecV1):
            self._spec = spec.__class__.from_string(spec.to_string())
        else:
            self._spec = DeviceSpec.from_string(spec)

    def __call__(self, node_def):
        current_device = DeviceSpec.from_string(node_def.device or '')
        return self._spec.make_merged_spec(current_device)

    def shortcut_string_merge(self, node_def):
        """Merge a node def without materializing a full DeviceSpec object.

    Often a device merge is invoked in order to generate a string which can be
    passed into the c api. In such a case, we can cache the
      node_def.device  ->  merge_result_string

    map, and in most cases avoid:
      - Materializing a copy of self._spec (In the case of DeviceSpecV1)
      - Materializing a DeviceSpec for node_def.device
      - A DeviceSpec.merge_from invocation

    In practice the cache hit rate for this function is very high, because the
    number of invocations when iterating through the device stack is much
    larger than the number of devices.

    Args:
      node_def: An Operation (or Operation-like) to merge device constraints
        with self._spec

    Returns:
      A string containing the merged device specification.
    """
        device = node_def.device or ''
        merge_key = (self._spec, device)
        result = _string_merge_cache.get(merge_key)
        if result is None:
            result = self.__call__(node_def).to_string()
            _string_merge_cache[merge_key] = result
        return result

    def __repr__(self):
        return '{} (spec: {})'.format(super(MergeDevice, self).__repr__(), self._spec.to_string())

    @property
    def is_null_merge(self):
        """Indicate whether the wrapped spec is empty.

    In the degenerate case where self._spec is an empty specification, a caller
    may wish to skip a merge step entirely. (However this class does not have
    enough information to make that determination.)

    Returns:
      A boolean indicating whether a device merge will be trivial.
    """
        return not bool(self._spec.to_string())