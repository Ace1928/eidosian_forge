from __future__ import annotations
import inspect
import random
import socket
import struct
from io import BytesIO
from itertools import chain
from typing import Optional, Sequence, SupportsInt, Union, overload
from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import CannotListenError
from twisted.python import failure, log, randbytes, util as tputil
from twisted.python.compat import cmp, comparable, nativeString
from twisted.names.error import (
def _nameToLabels(name):
    """
    Split a domain name into its constituent labels.

    @type name: L{bytes}
    @param name: A fully qualified domain name (with or without a
        trailing dot).

    @return: A L{list} of labels ending with an empty label
        representing the DNS root zone.
    @rtype: L{list} of L{bytes}
    """
    if name in (b'', b'.'):
        return [b'']
    labels = name.split(b'.')
    if labels[-1] != b'':
        labels.append(b'')
    return labels