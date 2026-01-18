from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
class PSTokenizer(object):

    def __init__(self, buf=b'', encoding='ascii'):
        buf = tobytes(buf)
        self.buf = buf
        self.len = len(buf)
        self.pos = 0
        self.closed = False
        self.encoding = encoding

    def read(self, n=-1):
        """Read at most 'n' bytes from the buffer, or less if the read
        hits EOF before obtaining 'n' bytes.
        If 'n' is negative or omitted, read all data until EOF is reached.
        """
        if self.closed:
            raise ValueError('I/O operation on closed file')
        if n is None or n < 0:
            newpos = self.len
        else:
            newpos = min(self.pos + n, self.len)
        r = self.buf[self.pos:newpos]
        self.pos = newpos
        return r

    def close(self):
        if not self.closed:
            self.closed = True
            del self.buf, self.pos

    def getnexttoken(self, len=len, ps_special=ps_special, stringmatch=stringRE.match, hexstringmatch=hexstringRE.match, commentmatch=commentRE.match, endmatch=endofthingRE.match):
        self.skipwhite()
        if self.pos >= self.len:
            return (None, None)
        pos = self.pos
        buf = self.buf
        char = bytechr(byteord(buf[pos]))
        if char in ps_special:
            if char in b'{}[]':
                tokentype = 'do_special'
                token = char
            elif char == b'%':
                tokentype = 'do_comment'
                _, nextpos = commentmatch(buf, pos).span()
                token = buf[pos:nextpos]
            elif char == b'(':
                tokentype = 'do_string'
                m = stringmatch(buf, pos)
                if m is None:
                    raise PSTokenError('bad string at character %d' % pos)
                _, nextpos = m.span()
                token = buf[pos:nextpos]
            elif char == b'<':
                tokentype = 'do_hexstring'
                m = hexstringmatch(buf, pos)
                if m is None:
                    raise PSTokenError('bad hexstring at character %d' % pos)
                _, nextpos = m.span()
                token = buf[pos:nextpos]
            else:
                raise PSTokenError('bad token at character %d' % pos)
        else:
            if char == b'/':
                tokentype = 'do_literal'
                m = endmatch(buf, pos + 1)
            else:
                tokentype = ''
                m = endmatch(buf, pos)
            if m is None:
                raise PSTokenError('bad token at character %d' % pos)
            _, nextpos = m.span()
            token = buf[pos:nextpos]
        self.pos = pos + len(token)
        token = tostr(token, encoding=self.encoding)
        return (tokentype, token)

    def skipwhite(self, whitematch=skipwhiteRE.match):
        _, nextpos = whitematch(self.buf, self.pos).span()
        self.pos = nextpos

    def starteexec(self):
        self.pos = self.pos + 1
        self.dirtybuf = self.buf[self.pos:]
        self.buf, R = eexec.decrypt(self.dirtybuf, 55665)
        self.len = len(self.buf)
        self.pos = 4

    def stopeexec(self):
        if not hasattr(self, 'dirtybuf'):
            return
        self.buf = self.dirtybuf
        del self.dirtybuf