from __future__ import absolute_import
import contextlib
import ctypes
import errno
import os.path
import shutil
import socket
import ssl
import struct
import threading
import weakref
import six
from .. import util
from ..util.ssl_ import PROTOCOL_TLS_CLIENT
from ._securetransport.bindings import CoreFoundation, Security, SecurityConst
from ._securetransport.low_level import (
def _set_alpn_protocols(self, protocols):
    """
        Sets up the ALPN protocols on the context.
        """
    if not protocols:
        return
    protocols_arr = _create_cfstring_array(protocols)
    try:
        result = Security.SSLSetALPNProtocols(self.context, protocols_arr)
        _assert_no_error(result)
    finally:
        CoreFoundation.CFRelease(protocols_arr)