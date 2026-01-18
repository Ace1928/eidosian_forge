import os
import abc
import codecs
import errno
import stat
import sys
from _thread import allocate_lock as Lock
import io
from io import (__all__, SEEK_SET, SEEK_CUR, SEEK_END)
from _io import FileIO
class IncrementalNewlineDecoder(codecs.IncrementalDecoder):
    """Codec used when reading a file in universal newlines mode.  It wraps
    another incremental decoder, translating \\r\\n and \\r into \\n.  It also
    records the types of newlines encountered.  When used with
    translate=False, it ensures that the newline sequence is returned in
    one piece.
    """

    def __init__(self, decoder, translate, errors='strict'):
        codecs.IncrementalDecoder.__init__(self, errors=errors)
        self.translate = translate
        self.decoder = decoder
        self.seennl = 0
        self.pendingcr = False

    def decode(self, input, final=False):
        if self.decoder is None:
            output = input
        else:
            output = self.decoder.decode(input, final=final)
        if self.pendingcr and (output or final):
            output = '\r' + output
            self.pendingcr = False
        if output.endswith('\r') and (not final):
            output = output[:-1]
            self.pendingcr = True
        crlf = output.count('\r\n')
        cr = output.count('\r') - crlf
        lf = output.count('\n') - crlf
        self.seennl |= (lf and self._LF) | (cr and self._CR) | (crlf and self._CRLF)
        if self.translate:
            if crlf:
                output = output.replace('\r\n', '\n')
            if cr:
                output = output.replace('\r', '\n')
        return output

    def getstate(self):
        if self.decoder is None:
            buf = b''
            flag = 0
        else:
            buf, flag = self.decoder.getstate()
        flag <<= 1
        if self.pendingcr:
            flag |= 1
        return (buf, flag)

    def setstate(self, state):
        buf, flag = state
        self.pendingcr = bool(flag & 1)
        if self.decoder is not None:
            self.decoder.setstate((buf, flag >> 1))

    def reset(self):
        self.seennl = 0
        self.pendingcr = False
        if self.decoder is not None:
            self.decoder.reset()
    _LF = 1
    _CR = 2
    _CRLF = 4

    @property
    def newlines(self):
        return (None, '\n', '\r', ('\r', '\n'), '\r\n', ('\n', '\r\n'), ('\r', '\r\n'), ('\r', '\n', '\r\n'))[self.seennl]