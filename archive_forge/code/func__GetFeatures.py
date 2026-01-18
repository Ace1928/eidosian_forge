import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def _GetFeatures(self):
    if not self._features:
        self._LazyLoadOptions()
    return self._features