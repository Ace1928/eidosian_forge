import errno
import fcntl
import os
import platform
import struct
import warnings
from collections import namedtuple
from typing import Tuple
from zope.interface import Attribute, Interface, implementer
from constantly import FlagConstant, Flags
from incremental import Version
from twisted.internet import abstract, defer, error, interfaces, task
from twisted.pair import ethernet, raw
from twisted.python import log
from twisted.python.deprecate import deprecated
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import FancyEqMixin, FancyStrMixin
def _openTunnel(self, name, mode):
    """
        Open the named tunnel using the given mode.

        @param name: The name of the tunnel to open.
        @type name: L{bytes}

        @param mode: Flags from L{TunnelFlags} with exactly one of
            L{TunnelFlags.IFF_TUN} or L{TunnelFlags.IFF_TAP} set.

        @return: A L{_TunnelDescription} representing the newly opened tunnel.
        """
    flags = self._system.O_RDWR | self._system.O_CLOEXEC | self._system.O_NONBLOCK
    config = struct.pack('%dsH' % (_IFNAMSIZ,), name, mode.value)
    fileno = self._system.open(_TUN_KO_PATH, flags)
    result = self._system.ioctl(fileno, _TUNSETIFF, config)
    return _TunnelDescription(fileno, result[:_IFNAMSIZ].strip(b'\x00'))