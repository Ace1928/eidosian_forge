from tensorflow.python import tf2
from tensorflow.python.framework import device_spec
Indicate whether the wrapped spec is empty.

    In the degenerate case where self._spec is an empty specification, a caller
    may wish to skip a merge step entirely. (However this class does not have
    enough information to make that determination.)

    Returns:
      A boolean indicating whether a device merge will be trivial.
    