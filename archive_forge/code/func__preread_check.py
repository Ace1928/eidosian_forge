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
def _preread_check(self):
    if not self._read_buf:
        if not self._read_check_passed:
            raise errors.PermissionDeniedError(None, None, "File isn't open for reading")
        self._read_buf = _pywrap_file_io.BufferedInputStream(compat.path_to_str(self.__name), 1024 * 512)