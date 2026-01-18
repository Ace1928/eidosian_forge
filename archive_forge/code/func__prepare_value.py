import binascii
import os
from posixpath import join as urljoin
import uuid
import six
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import _pywrap_file_io
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _prepare_value(self, val):
    if self._binary_mode:
        return compat.as_bytes(val, encoding=self.__encoding)
    else:
        return compat.as_str_any(val, encoding=self.__encoding)