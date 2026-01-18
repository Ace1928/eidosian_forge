import ctypes
import enum
import os
import platform
import sys
import numpy as np
class Delegate:
    """Python wrapper class to manage TfLiteDelegate objects.

  The shared library is expected to have two functions,
  tflite_plugin_create_delegate and tflite_plugin_destroy_delegate,
  which should implement the API specified in
  tensorflow/lite/delegates/external/external_delegate_interface.h.
  """

    def __init__(self, library, options=None):
        """Loads delegate from the shared library.

    Args:
      library: Shared library name.
      options: Dictionary of options that are required to load the delegate. All
        keys and values in the dictionary should be serializable. Consult the
        documentation of the specific delegate for required and legal options.
        (default None)

    Raises:
      RuntimeError: This is raised if the Python implementation is not CPython.
    """
        if platform.python_implementation() != 'CPython':
            raise RuntimeError('Delegates are currently only supported into CPythondue to missing immediate reference counting.')
        self._library = ctypes.pydll.LoadLibrary(library)
        self._library.tflite_plugin_create_delegate.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.CFUNCTYPE(None, ctypes.c_char_p)]
        self._library.tflite_plugin_create_delegate.restype = ctypes.c_void_p
        options = options or {}
        options_keys = (ctypes.c_char_p * len(options))()
        options_values = (ctypes.c_char_p * len(options))()
        for idx, (key, value) in enumerate(options.items()):
            options_keys[idx] = str(key).encode('utf-8')
            options_values[idx] = str(value).encode('utf-8')

        class ErrorMessageCapture:

            def __init__(self):
                self.message = ''

            def report(self, x):
                self.message += x if isinstance(x, str) else x.decode('utf-8')
        capture = ErrorMessageCapture()
        error_capturer_cb = ctypes.CFUNCTYPE(None, ctypes.c_char_p)(capture.report)
        self._delegate_ptr = self._library.tflite_plugin_create_delegate(options_keys, options_values, len(options), error_capturer_cb)
        if self._delegate_ptr is None:
            raise ValueError(capture.message)

    def __del__(self):
        if self._library is not None:
            self._library.tflite_plugin_destroy_delegate.argtypes = [ctypes.c_void_p]
            self._library.tflite_plugin_destroy_delegate(self._delegate_ptr)
            self._library = None

    def _get_native_delegate_pointer(self):
        """Returns the native TfLiteDelegate pointer.

    It is not safe to copy this pointer because it needs to be freed.

    Returns:
      TfLiteDelegate *
    """
        return self._delegate_ptr