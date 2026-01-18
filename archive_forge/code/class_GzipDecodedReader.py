from __future__ import (absolute_import, division, print_function)
import atexit
import base64
import email.mime.multipart
import email.mime.nonmultipart
import email.mime.application
import email.parser
import email.utils
import functools
import io
import mimetypes
import netrc
import os
import platform
import re
import socket
import sys
import tempfile
import traceback
import types  # pylint: disable=unused-import
from contextlib import contextmanager
import ansible.module_utils.compat.typing as t
import ansible.module_utils.six.moves.http_cookiejar as cookiejar
import ansible.module_utils.six.moves.urllib.error as urllib_error
from ansible.module_utils.common.collections import Mapping, is_sequence
from ansible.module_utils.six import PY2, PY3, string_types
from ansible.module_utils.six.moves import cStringIO
from ansible.module_utils.basic import get_distribution, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
class GzipDecodedReader(GzipFile):
    """A file-like object to decode a response encoded with the gzip
    method, as described in RFC 1952.

    Largely copied from ``xmlrpclib``/``xmlrpc.client``
    """

    def __init__(self, fp):
        if not HAS_GZIP:
            raise MissingModuleError(self.missing_gzip_error(), import_traceback=GZIP_IMP_ERR)
        if PY3:
            self._io = fp
        else:
            self._io = io.BytesIO()
            for block in iter(functools.partial(fp.read, 65536), b''):
                self._io.write(block)
            self._io.seek(0)
            fp.close()
        gzip.GzipFile.__init__(self, mode='rb', fileobj=self._io)

    def close(self):
        try:
            gzip.GzipFile.close(self)
        finally:
            self._io.close()

    @staticmethod
    def missing_gzip_error():
        return missing_required_lib('gzip', reason='to decompress gzip encoded responses. Set "decompress" to False, to prevent attempting auto decompression')