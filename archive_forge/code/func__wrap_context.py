import errno
import os
import re
import socket
import ssl
from contextlib import contextmanager
from ssl import SSLError
from struct import pack, unpack
from .exceptions import UnexpectedFrame
from .platform import KNOWN_TCP_OPTS, SOL_TCP
from .utils import set_cloexec
def _wrap_context(self, sock, sslopts, check_hostname=None, **ctx_options):
    """Wrap socket without SNI headers.

        PARAMETERS:
            sock: socket.socket

            Socket to be wrapped.

            sslopts: dict

                Parameters of  :attr:`ssl.SSLContext.wrap_socket`.

            check_hostname

                Whether to match the peer cert’s hostname. See
                :attr:`ssl.SSLContext.check_hostname` for details.

            ctx_options

                Parameters of :attr:`ssl.create_default_context`.
        """
    ctx = ssl.create_default_context(**ctx_options)
    ctx.check_hostname = check_hostname
    return ctx.wrap_socket(sock, **sslopts)