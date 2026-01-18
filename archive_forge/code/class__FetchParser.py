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
class _FetchParser:

    class Envelope:
        type = 'envelope'
        __str__ = lambda self: 'envelope'

    class Flags:
        type = 'flags'
        __str__ = lambda self: 'flags'

    class InternalDate:
        type = 'internaldate'
        __str__ = lambda self: 'internaldate'

    class RFC822Header:
        type = 'rfc822header'
        __str__ = lambda self: 'rfc822.header'

    class RFC822Text:
        type = 'rfc822text'
        __str__ = lambda self: 'rfc822.text'

    class RFC822Size:
        type = 'rfc822size'
        __str__ = lambda self: 'rfc822.size'

    class RFC822:
        type = 'rfc822'
        __str__ = lambda self: 'rfc822'

    class UID:
        type = 'uid'
        __str__ = lambda self: 'uid'

    class Body:
        type = 'body'
        peek = False
        header = None
        mime = None
        text = None
        part = ()
        empty = False
        partialBegin = None
        partialLength = None

        def __str__(self) -> str:
            return self.__bytes__().decode('ascii')

        def __bytes__(self) -> bytes:
            base = b'BODY'
            part = b''
            separator = b''
            if self.part:
                part = b'.'.join([str(x + 1).encode('ascii') for x in self.part])
                separator = b'.'
            if self.header:
                base += b'[' + part + separator + str(self.header).encode('ascii') + b']'
            elif self.text:
                base += b'[' + part + separator + b'TEXT]'
            elif self.mime:
                base += b'[' + part + separator + b'MIME]'
            elif self.empty:
                base += b'[' + part + b']'
            if self.partialBegin is not None:
                base += b'<%d.%d>' % (self.partialBegin, self.partialLength)
            return base

    class BodyStructure:
        type = 'bodystructure'
        __str__ = lambda self: 'bodystructure'

    class Header:
        negate = False
        fields = None
        part = None

        def __str__(self) -> str:
            return self.__bytes__().decode('ascii')

        def __bytes__(self) -> bytes:
            base = b'HEADER'
            if self.fields:
                base += b'.FIELDS'
                if self.negate:
                    base += b'.NOT'
                fields = []
                for f in self.fields:
                    f = f.title()
                    if _needsQuote(f):
                        f = _quote(f)
                    fields.append(f)
                base += b' (' + b' '.join(fields) + b')'
            if self.part:
                base = b'.'.join([(x + 1).__bytes__() for x in self.part]) + b'.' + base
            return base

    class Text:
        pass

    class MIME:
        pass
    parts = None
    _simple_fetch_att = [(b'envelope', Envelope), (b'flags', Flags), (b'internaldate', InternalDate), (b'rfc822.header', RFC822Header), (b'rfc822.text', RFC822Text), (b'rfc822.size', RFC822Size), (b'rfc822', RFC822), (b'uid', UID), (b'bodystructure', BodyStructure)]

    def __init__(self):
        self.state = ['initial']
        self.result = []
        self.remaining = b''

    def parseString(self, s):
        s = self.remaining + s
        try:
            while s or self.state:
                if not self.state:
                    raise IllegalClientResponse('Invalid Argument')
                state = self.state.pop()
                try:
                    used = getattr(self, 'state_' + state)(s)
                except BaseException:
                    self.state.append(state)
                    raise
                else:
                    s = s[used:]
        finally:
            self.remaining = s

    def state_initial(self, s):
        if s == b'':
            return 0
        l = s.lower()
        if l.startswith(b'all'):
            self.result.extend((self.Flags(), self.InternalDate(), self.RFC822Size(), self.Envelope()))
            return 3
        if l.startswith(b'full'):
            self.result.extend((self.Flags(), self.InternalDate(), self.RFC822Size(), self.Envelope(), self.Body()))
            return 4
        if l.startswith(b'fast'):
            self.result.extend((self.Flags(), self.InternalDate(), self.RFC822Size()))
            return 4
        if l.startswith(b'('):
            self.state.extend(('close_paren', 'maybe_fetch_att', 'fetch_att'))
            return 1
        self.state.append('fetch_att')
        return 0

    def state_close_paren(self, s):
        if s.startswith(b')'):
            return 1
        raise Exception('Missing )')

    def state_whitespace(self, s):
        if not s or not s[0:1].isspace():
            raise Exception('Whitespace expected, none found')
        i = 0
        for i in range(len(s)):
            if not s[i:i + 1].isspace():
                break
        return i

    def state_maybe_fetch_att(self, s):
        if not s.startswith(b')'):
            self.state.extend(('maybe_fetch_att', 'fetch_att', 'whitespace'))
        return 0

    def state_fetch_att(self, s):
        l = s.lower()
        for name, cls in self._simple_fetch_att:
            if l.startswith(name):
                self.result.append(cls())
                return len(name)
        b = self.Body()
        if l.startswith(b'body.peek'):
            b.peek = True
            used = 9
        elif l.startswith(b'body'):
            used = 4
        else:
            raise Exception(f'Nothing recognized in fetch_att: {l}')
        self.pending_body = b
        self.state.extend(('got_body', 'maybe_partial', 'maybe_section'))
        return used

    def state_got_body(self, s):
        self.result.append(self.pending_body)
        del self.pending_body
        return 0

    def state_maybe_section(self, s):
        if not s.startswith(b'['):
            return 0
        self.state.extend(('section', 'part_number'))
        return 1
    _partExpr = re.compile(b'(\\d+(?:\\.\\d+)*)\\.?')

    def state_part_number(self, s):
        m = self._partExpr.match(s)
        if m is not None:
            self.parts = [int(p) - 1 for p in m.groups()[0].split(b'.')]
            return m.end()
        else:
            self.parts = []
            return 0

    def state_section(self, s):
        l = s.lower()
        used = 0
        if l.startswith(b']'):
            self.pending_body.empty = True
            used += 1
        elif l.startswith(b'header]'):
            h = self.pending_body.header = self.Header()
            h.negate = True
            h.fields = ()
            used += 7
        elif l.startswith(b'text]'):
            self.pending_body.text = self.Text()
            used += 5
        elif l.startswith(b'mime]'):
            self.pending_body.mime = self.MIME()
            used += 5
        else:
            h = self.Header()
            if l.startswith(b'header.fields.not'):
                h.negate = True
                used += 17
            elif l.startswith(b'header.fields'):
                used += 13
            else:
                raise Exception(f'Unhandled section contents: {l!r}')
            self.pending_body.header = h
            self.state.extend(('finish_section', 'header_list', 'whitespace'))
        self.pending_body.part = tuple(self.parts)
        self.parts = None
        return used

    def state_finish_section(self, s):
        if not s.startswith(b']'):
            raise Exception('section must end with ]')
        return 1

    def state_header_list(self, s):
        if not s.startswith(b'('):
            raise Exception('Header list must begin with (')
        end = s.find(b')')
        if end == -1:
            raise Exception('Header list must end with )')
        headers = s[1:end].split()
        self.pending_body.header.fields = [h.upper() for h in headers]
        return end + 1

    def state_maybe_partial(self, s):
        if not s.startswith(b'<'):
            return 0
        end = s.find(b'>')
        if end == -1:
            raise Exception('Found < but not >')
        partial = s[1:end]
        parts = partial.split(b'.', 1)
        if len(parts) != 2:
            raise Exception('Partial specification did not include two .-delimited integers')
        begin, length = map(int, parts)
        self.pending_body.partialBegin = begin
        self.pending_body.partialLength = length
        return end + 1