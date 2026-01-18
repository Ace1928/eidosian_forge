import _thread
import errno
import io
import os
import sys
import time
import breezy
from ...lazy_import import lazy_import
import select
import socket
import weakref
from breezy import (
from breezy.i18n import gettext
from breezy.bzr.smart import client, protocol, request, signals, vfs
from breezy.transport import ssh
from ... import errors, osutils
def _get_protocol_factory_for_bytes(bytes):
    """Determine the right protocol factory for 'bytes'.

    This will return an appropriate protocol factory depending on the version
    of the protocol being used, as determined by inspecting the given bytes.
    The bytes should have at least one newline byte (i.e. be a whole line),
    otherwise it's possible that a request will be incorrectly identified as
    version 1.

    Typical use would be::

         factory, unused_bytes = _get_protocol_factory_for_bytes(bytes)
         server_protocol = factory(transport, write_func, root_client_path)
         server_protocol.accept_bytes(unused_bytes)

    :param bytes: a str of bytes of the start of the request.
    :returns: 2-tuple of (protocol_factory, unused_bytes).  protocol_factory is
        a callable that takes three args: transport, write_func,
        root_client_path.  unused_bytes are any bytes that were not part of a
        protocol version marker.
    """
    if bytes.startswith(protocol.MESSAGE_VERSION_THREE):
        protocol_factory = protocol.build_server_protocol_three
        bytes = bytes[len(protocol.MESSAGE_VERSION_THREE):]
    elif bytes.startswith(protocol.REQUEST_VERSION_TWO):
        protocol_factory = protocol.SmartServerRequestProtocolTwo
        bytes = bytes[len(protocol.REQUEST_VERSION_TWO):]
    else:
        protocol_factory = protocol.SmartServerRequestProtocolOne
    return (protocol_factory, bytes)