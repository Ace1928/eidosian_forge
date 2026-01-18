import threading
import warnings
from cloudsdk.google.protobuf.internal import api_implementation
def _SetOptions(self, options, options_class_name):
    """Sets the descriptor's options

    This function is used in generated proto2 files to update descriptor
    options. It must not be used outside proto2.
    """
    self._options = options
    self._options_class_name = options_class_name
    self.has_options = options is not None