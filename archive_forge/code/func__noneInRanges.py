import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import (
def _noneInRanges(self):
    """
        Is there a L{None} in our ranges?

        L{MessageSet.clean} merges overlapping or consecutive ranges.
        None is represents a value larger than any number.  There are
        thus two cases:

            1. C{(x, *) + (y, z)} such that C{x} is smaller than C{y}

            2. C{(z, *) + (x, y)} such that C{z} is larger than C{y}

        (Other cases, such as C{y < x < z}, can be split into these
        two cases; for example C{(y - 1, y)} + C{(x, x) + (z, z + 1)})

        In case 1, C{* > y} and C{* > z}, so C{(x, *) + (y, z) = (x,
        *)}

        In case 2, C{z > x and z > y}, so the intervals do not merge,
        and the ranges are sorted as C{[(x, y), (z, *)]}.  C{*} is
        represented as C{(*, *)}, so this is the same as 2.  but with
        a C{z} that is greater than everything.

        The result is that there is a maximum of two L{None}s, and one
        of them has to be the high element in the last tuple in
        C{self.ranges}.  That means checking if C{self.ranges[-1][-1]}
        is L{None} suffices to check if I{any} element is L{None}.

        @return: L{True} if L{None} is in some range in ranges and
            L{False} if otherwise.
        """
    return self.ranges[-1][-1] is None