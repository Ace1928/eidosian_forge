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
def _cbCopySelectedMailbox(self, mbox, tag, messages, mailbox, uid):
    if not mbox:
        self.sendNegativeResponse(tag, 'No such mailbox: ' + mailbox)
    else:
        maybeDeferred(self.mbox.fetch, messages, uid).addCallback(self.__cbCopy, tag, mbox).addCallback(self.__cbCopied, tag, mbox).addErrback(self.__ebCopy, tag)