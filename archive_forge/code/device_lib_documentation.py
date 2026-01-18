from tensorflow.core.framework import device_attributes_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import _pywrap_device_lib
List the available devices available in the local process.

  Args:
    session_config: a session config proto or None to use the default config.

  Returns:
    A list of `DeviceAttribute` protocol buffers.
  