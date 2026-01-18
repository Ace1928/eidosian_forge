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
@implementer(IAccount)
class MemoryAccountWithoutNamespaces:
    mailboxes = None
    subscriptions = None
    top_id = 0

    def __init__(self, name):
        self.name = name
        self.mailboxes = {}
        self.subscriptions = []

    def allocateID(self):
        id = self.top_id
        self.top_id += 1
        return id

    def addMailbox(self, name, mbox=None):
        name = _parseMbox(name.upper())
        if name in self.mailboxes:
            raise MailboxCollision(name)
        if mbox is None:
            mbox = self._emptyMailbox(name, self.allocateID())
        self.mailboxes[name] = mbox
        return 1

    def create(self, pathspec):
        paths = [path for path in pathspec.split('/') if path]
        for accum in range(1, len(paths)):
            try:
                self.addMailbox('/'.join(paths[:accum]))
            except MailboxCollision:
                pass
        try:
            self.addMailbox('/'.join(paths))
        except MailboxCollision:
            if not pathspec.endswith('/'):
                return False
        return True

    def _emptyMailbox(self, name, id):
        raise NotImplementedError

    def select(self, name, readwrite=1):
        return self.mailboxes.get(_parseMbox(name.upper()))

    def delete(self, name):
        name = _parseMbox(name.upper())
        mbox = self.mailboxes.get(name)
        if not mbox:
            raise MailboxException('No such mailbox')
        if '\\Noselect' in mbox.getFlags():
            for others in self.mailboxes.keys():
                if others != name and others.startswith(name):
                    raise MailboxException('Hierarchically inferior mailboxes exist and \\Noselect is set')
        mbox.destroy()
        if len(self._inferiorNames(name)) > 1:
            raise MailboxException(f'Name "{name}" has inferior hierarchical names')
        del self.mailboxes[name]

    def rename(self, oldname, newname):
        oldname = _parseMbox(oldname.upper())
        newname = _parseMbox(newname.upper())
        if oldname not in self.mailboxes:
            raise NoSuchMailbox(oldname)
        inferiors = self._inferiorNames(oldname)
        inferiors = [(o, o.replace(oldname, newname, 1)) for o in inferiors]
        for old, new in inferiors:
            if new in self.mailboxes:
                raise MailboxCollision(new)
        for old, new in inferiors:
            self.mailboxes[new] = self.mailboxes[old]
            del self.mailboxes[old]

    def _inferiorNames(self, name):
        inferiors = []
        for infname in self.mailboxes.keys():
            if infname.startswith(name):
                inferiors.append(infname)
        return inferiors

    def isSubscribed(self, name):
        return _parseMbox(name.upper()) in self.subscriptions

    def subscribe(self, name):
        name = _parseMbox(name.upper())
        if name not in self.subscriptions:
            self.subscriptions.append(name)

    def unsubscribe(self, name):
        name = _parseMbox(name.upper())
        if name not in self.subscriptions:
            raise MailboxException(f'Not currently subscribed to {name}')
        self.subscriptions.remove(name)

    def listMailboxes(self, ref, wildcard):
        ref = self._inferiorNames(_parseMbox(ref.upper()))
        wildcard = wildcardToRegexp(wildcard, '/')
        return [(i, self.mailboxes[i]) for i in ref if wildcard.match(i)]