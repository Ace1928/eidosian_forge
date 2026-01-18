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
def _responseFromMessage(responseConstructor, message, **kwargs):
    """
    Generate a L{Message} like instance suitable for use as the response to
    C{message}.

    The C{queries}, C{id} attributes will be copied from C{message} and the
    C{answer} flag will be set to L{True}.

    @param responseConstructor: A response message constructor with an
         initializer signature matching L{dns.Message.__init__}.
    @type responseConstructor: C{callable}

    @param message: A request message.
    @type message: L{Message}

    @param kwargs: Keyword arguments which will be passed to the initialiser
        of the response message.
    @type kwargs: L{dict}

    @return: A L{Message} like response instance.
    @rtype: C{responseConstructor}
    """
    response = responseConstructor(id=message.id, answer=True, **kwargs)
    response.queries = message.queries[:]
    return response