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
def _cbSelectWork(self, mbox, cmdName, tag):
    if mbox is None:
        self.sendNegativeResponse(tag, b'No such mailbox')
        return
    if '\\noselect' in [s.lower() for s in mbox.getFlags()]:
        self.sendNegativeResponse(tag, 'Mailbox cannot be selected')
        return
    flags = [networkString(flag) for flag in mbox.getFlags()]
    self.sendUntaggedResponse(b'%d EXISTS' % (mbox.getMessageCount(),))
    self.sendUntaggedResponse(b'%d RECENT' % (mbox.getRecentCount(),))
    self.sendUntaggedResponse(b'FLAGS (' + b' '.join(flags) + b')')
    self.sendPositiveResponse(None, b'[UIDVALIDITY %d]' % (mbox.getUIDValidity(),))
    s = mbox.isWriteable() and b'READ-WRITE' or b'READ-ONLY'
    mbox.addListener(self)
    self.sendPositiveResponse(tag, b'[' + s + b'] ' + cmdName + b' successful')
    self.state = 'select'
    self.mbox = mbox