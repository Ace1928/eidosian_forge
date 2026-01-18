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
def _disposition(self, disp):
    """
        Parse a I{Content-Disposition} header into a two-sequence of the
        disposition and a flattened list of its parameters.

        @return: L{None} if there is no disposition header value, a L{list} with
            two elements otherwise.
        """
    if disp:
        disp = disp.split('; ')
        if len(disp) == 1:
            disp = (disp[0].lower(), None)
        elif len(disp) > 1:
            params = [x for param in disp[1:] for x in param.split('=', 1)]
            disp = [disp[0].lower(), params]
        return disp
    else:
        return None