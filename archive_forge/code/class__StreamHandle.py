import io
import logging
import os
import sys
import threading
import time
from io import StringIO
class _StreamHandle(object):

    def __init__(self, mode, buffering, encoding, newline):
        self.buffering = buffering
        self.newlines = newline
        self.read_pipe, self.write_pipe = os.pipe()
        if not buffering and 'b' not in mode:
            buffering = -1
        self.write_file = os.fdopen(self.write_pipe, mode=mode, buffering=buffering, encoding=encoding, newline=newline, closefd=False)
        self.decoder_buffer = b''
        try:
            self.encoding = encoding or self.write_file.encoding
        except AttributeError:
            self.encoding = None
        if self.encoding:
            self.output_buffer = ''
        else:
            self.output_buffer = b''

    def __repr__(self):
        return '%s(%s)' % (self.buffering, id(self))

    def fileno(self):
        return self.read_pipe

    def close(self):
        if self.write_file is not None:
            self.write_file.close()
            self.write_file = None
        if self.write_pipe is not None:
            try:
                os.close(self.write_pipe)
            except OSError:
                pass
            self.write_pipe = None

    def finalize(self, ostreams):
        self.decodeIncomingBuffer()
        if ostreams:
            self.buffering = 0
            self.writeOutputBuffer(ostreams)
        os.close(self.read_pipe)
        if self.output_buffer:
            logger.error("Stream handle closed with a partial line in the output buffer that was not emitted to the output stream(s):\n\t'%s'" % (self.output_buffer,))
        if self.decoder_buffer:
            logger.error('Stream handle closed with un-decoded characters in the decoder buffer that was not emitted to the output stream(s):\n\t%r' % (self.decoder_buffer,))

    def decodeIncomingBuffer(self):
        if not self.encoding:
            self.output_buffer, self.decoder_buffer = (self.decoder_buffer, b'')
            return
        raw_len = len(self.decoder_buffer)
        chars = ''
        while raw_len:
            try:
                chars = self.decoder_buffer[:raw_len].decode(self.encoding)
                break
            except:
                pass
            raw_len -= 1
        if self.newlines is None:
            chars = chars.replace('\r\n', '\n').replace('\r', '\n')
        self.output_buffer += chars
        self.decoder_buffer = self.decoder_buffer[raw_len:]

    def writeOutputBuffer(self, ostreams):
        if not self.encoding:
            ostring, self.output_buffer = (self.output_buffer, b'')
        elif self.buffering == 1:
            EOL = self.output_buffer.rfind(self.newlines or '\n') + 1
            ostring = self.output_buffer[:EOL]
            self.output_buffer = self.output_buffer[EOL:]
        else:
            ostring, self.output_buffer = (self.output_buffer, '')
        if not ostring:
            return
        for stream in ostreams:
            try:
                written = stream.write(ostring)
            except:
                written = 0
            if written and (not self.buffering):
                stream.flush()
            if written is not None and written != len(ostring):
                logger.error('Output stream (%s) closed before all output was written to it. The following was left in the output buffer:\n\t%r' % (stream, ostring[written:]))