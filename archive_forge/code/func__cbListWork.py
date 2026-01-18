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
def _cbListWork(self, mailboxes, tag, sub, cmdName):
    for name, box in mailboxes:
        if not sub or self.account.isSubscribed(name):
            flags = [networkString(flag) for flag in box.getFlags()]
            delim = box.getHierarchicalDelimiter().encode('imap4-utf-7')
            resp = (DontQuoteMe(cmdName), map(DontQuoteMe, flags), delim, name.encode('imap4-utf-7'))
            self.sendUntaggedResponse(collapseNestedLists(resp))
    self.sendPositiveResponse(tag, cmdName + b' completed')