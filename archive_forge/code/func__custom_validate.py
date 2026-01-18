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
def _custom_validate(self, verify, trust_bundle):
    """
        Called when we have set custom validation. We do this in two cases:
        first, when cert validation is entirely disabled; and second, when
        using a custom trust DB.
        Raises an SSLError if the connection is not trusted.
        """
    if not verify:
        return
    successes = (SecurityConst.kSecTrustResultUnspecified, SecurityConst.kSecTrustResultProceed)
    try:
        trust_result = self._evaluate_trust(trust_bundle)
        if trust_result in successes:
            return
        reason = 'error code: %d' % (trust_result,)
    except Exception as e:
        reason = 'exception: %r' % (e,)
    rec = _build_tls_unknown_ca_alert(self.version())
    self.socket.sendall(rec)
    opts = struct.pack('ii', 1, 0)
    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, opts)
    self.close()
    raise ssl.SSLError('certificate verify failed, %s' % reason)