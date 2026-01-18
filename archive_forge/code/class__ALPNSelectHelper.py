import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
class _ALPNSelectHelper(_CallbackExceptionHelper):
    """
    Wrap a callback such that it can be used as an ALPN selection callback.
    """

    def __init__(self, callback):
        _CallbackExceptionHelper.__init__(self)

        @wraps(callback)
        def wrapper(ssl, out, outlen, in_, inlen, arg):
            try:
                conn = Connection._reverse_mapping[ssl]
                instr = _ffi.buffer(in_, inlen)[:]
                protolist = []
                while instr:
                    encoded_len = instr[0]
                    proto = instr[1:encoded_len + 1]
                    protolist.append(proto)
                    instr = instr[encoded_len + 1:]
                outbytes = callback(conn, protolist)
                any_accepted = True
                if outbytes is NO_OVERLAPPING_PROTOCOLS:
                    outbytes = b''
                    any_accepted = False
                elif not isinstance(outbytes, bytes):
                    raise TypeError('ALPN callback must return a bytestring or the special NO_OVERLAPPING_PROTOCOLS sentinel value.')
                conn._alpn_select_callback_args = [_ffi.new('unsigned char *', len(outbytes)), _ffi.new('unsigned char[]', outbytes)]
                outlen[0] = conn._alpn_select_callback_args[0][0]
                out[0] = conn._alpn_select_callback_args[1]
                if not any_accepted:
                    return _lib.SSL_TLSEXT_ERR_NOACK
                return _lib.SSL_TLSEXT_ERR_OK
            except Exception as e:
                self._problems.append(e)
                return _lib.SSL_TLSEXT_ERR_ALERT_FATAL
        self.callback = _ffi.callback('int (*)(SSL *, unsigned char **, unsigned char *, const unsigned char *, unsigned int, void *)', wrapper)