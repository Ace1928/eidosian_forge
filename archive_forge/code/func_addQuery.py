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
def addQuery(self, name, type=ALL_RECORDS, cls=IN):
    """
        Add another query to this Message.

        @type name: L{bytes}
        @param name: The name to query.

        @type type: L{int}
        @param type: Query type

        @type cls: L{int}
        @param cls: Query class
        """
    self.queries.append(Query(name, type, cls))