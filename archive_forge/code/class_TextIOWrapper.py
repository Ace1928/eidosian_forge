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
class TextIOWrapper(TextIOBase):
    """Character and line based layer over a BufferedIOBase object, buffer.

    encoding gives the name of the encoding that the stream will be
    decoded or encoded with. It defaults to locale.getencoding().

    errors determines the strictness of encoding and decoding (see the
    codecs.register) and defaults to "strict".

    newline can be None, '', '\\n', '\\r', or '\\r\\n'.  It controls the
    handling of line endings. If it is None, universal newlines is
    enabled.  With this enabled, on input, the lines endings '\\n', '\\r',
    or '\\r\\n' are translated to '\\n' before being returned to the
    caller. Conversely, on output, '\\n' is translated to the system
    default line separator, os.linesep. If newline is any other of its
    legal values, that newline becomes the newline when the file is read
    and it is returned untranslated. On output, '\\n' is converted to the
    newline.

    If line_buffering is True, a call to flush is implied when a call to
    write contains a newline character.
    """
    _CHUNK_SIZE = 2048
    _buffer = None

    def __init__(self, buffer, encoding=None, errors=None, newline=None, line_buffering=False, write_through=False):
        self._check_newline(newline)
        encoding = text_encoding(encoding)
        if encoding == 'locale':
            encoding = self._get_locale_encoding()
        if not isinstance(encoding, str):
            raise ValueError('invalid encoding: %r' % encoding)
        if not codecs.lookup(encoding)._is_text_encoding:
            msg = '%r is not a text encoding; use codecs.open() to handle arbitrary codecs'
            raise LookupError(msg % encoding)
        if errors is None:
            errors = 'strict'
        else:
            if not isinstance(errors, str):
                raise ValueError('invalid errors: %r' % errors)
            if _CHECK_ERRORS:
                codecs.lookup_error(errors)
        self._buffer = buffer
        self._decoded_chars = ''
        self._decoded_chars_used = 0
        self._snapshot = None
        self._seekable = self._telling = self.buffer.seekable()
        self._has_read1 = hasattr(self.buffer, 'read1')
        self._configure(encoding, errors, newline, line_buffering, write_through)

    def _check_newline(self, newline):
        if newline is not None and (not isinstance(newline, str)):
            raise TypeError('illegal newline type: %r' % (type(newline),))
        if newline not in (None, '', '\n', '\r', '\r\n'):
            raise ValueError('illegal newline value: %r' % (newline,))

    def _configure(self, encoding=None, errors=None, newline=None, line_buffering=False, write_through=False):
        self._encoding = encoding
        self._errors = errors
        self._encoder = None
        self._decoder = None
        self._b2cratio = 0.0
        self._readuniversal = not newline
        self._readtranslate = newline is None
        self._readnl = newline
        self._writetranslate = newline != ''
        self._writenl = newline or os.linesep
        self._line_buffering = line_buffering
        self._write_through = write_through
        if self._seekable and self.writable():
            position = self.buffer.tell()
            if position != 0:
                try:
                    self._get_encoder().setstate(0)
                except LookupError:
                    pass

    def __repr__(self):
        result = '<{}.{}'.format(self.__class__.__module__, self.__class__.__qualname__)
        try:
            name = self.name
        except AttributeError:
            pass
        else:
            result += ' name={0!r}'.format(name)
        try:
            mode = self.mode
        except AttributeError:
            pass
        else:
            result += ' mode={0!r}'.format(mode)
        return result + ' encoding={0!r}>'.format(self.encoding)

    @property
    def encoding(self):
        return self._encoding

    @property
    def errors(self):
        return self._errors

    @property
    def line_buffering(self):
        return self._line_buffering

    @property
    def write_through(self):
        return self._write_through

    @property
    def buffer(self):
        return self._buffer

    def reconfigure(self, *, encoding=None, errors=None, newline=Ellipsis, line_buffering=None, write_through=None):
        """Reconfigure the text stream with new parameters.

        This also flushes the stream.
        """
        if self._decoder is not None and (encoding is not None or errors is not None or newline is not Ellipsis):
            raise UnsupportedOperation('It is not possible to set the encoding or newline of stream after the first read')
        if errors is None:
            if encoding is None:
                errors = self._errors
            else:
                errors = 'strict'
        elif not isinstance(errors, str):
            raise TypeError('invalid errors: %r' % errors)
        if encoding is None:
            encoding = self._encoding
        else:
            if not isinstance(encoding, str):
                raise TypeError('invalid encoding: %r' % encoding)
            if encoding == 'locale':
                encoding = self._get_locale_encoding()
        if newline is Ellipsis:
            newline = self._readnl
        self._check_newline(newline)
        if line_buffering is None:
            line_buffering = self.line_buffering
        if write_through is None:
            write_through = self.write_through
        self.flush()
        self._configure(encoding, errors, newline, line_buffering, write_through)

    def seekable(self):
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        return self._seekable

    def readable(self):
        return self.buffer.readable()

    def writable(self):
        return self.buffer.writable()

    def flush(self):
        self.buffer.flush()
        self._telling = self._seekable

    def close(self):
        if self.buffer is not None and (not self.closed):
            try:
                self.flush()
            finally:
                self.buffer.close()

    @property
    def closed(self):
        return self.buffer.closed

    @property
    def name(self):
        return self.buffer.name

    def fileno(self):
        return self.buffer.fileno()

    def isatty(self):
        return self.buffer.isatty()

    def write(self, s):
        """Write data, where s is a str"""
        if self.closed:
            raise ValueError('write to closed file')
        if not isinstance(s, str):
            raise TypeError("can't write %s to text stream" % s.__class__.__name__)
        length = len(s)
        haslf = (self._writetranslate or self._line_buffering) and '\n' in s
        if haslf and self._writetranslate and (self._writenl != '\n'):
            s = s.replace('\n', self._writenl)
        encoder = self._encoder or self._get_encoder()
        b = encoder.encode(s)
        self.buffer.write(b)
        if self._line_buffering and (haslf or '\r' in s):
            self.flush()
        if self._snapshot is not None:
            self._set_decoded_chars('')
            self._snapshot = None
        if self._decoder:
            self._decoder.reset()
        return length

    def _get_encoder(self):
        make_encoder = codecs.getincrementalencoder(self._encoding)
        self._encoder = make_encoder(self._errors)
        return self._encoder

    def _get_decoder(self):
        make_decoder = codecs.getincrementaldecoder(self._encoding)
        decoder = make_decoder(self._errors)
        if self._readuniversal:
            decoder = IncrementalNewlineDecoder(decoder, self._readtranslate)
        self._decoder = decoder
        return decoder

    def _set_decoded_chars(self, chars):
        """Set the _decoded_chars buffer."""
        self._decoded_chars = chars
        self._decoded_chars_used = 0

    def _get_decoded_chars(self, n=None):
        """Advance into the _decoded_chars buffer."""
        offset = self._decoded_chars_used
        if n is None:
            chars = self._decoded_chars[offset:]
        else:
            chars = self._decoded_chars[offset:offset + n]
        self._decoded_chars_used += len(chars)
        return chars

    def _get_locale_encoding(self):
        try:
            import locale
        except ImportError:
            return 'utf-8'
        else:
            return locale.getencoding()

    def _rewind_decoded_chars(self, n):
        """Rewind the _decoded_chars buffer."""
        if self._decoded_chars_used < n:
            raise AssertionError('rewind decoded_chars out of bounds')
        self._decoded_chars_used -= n

    def _read_chunk(self):
        """
        Read and decode the next chunk of data from the BufferedReader.
        """
        if self._decoder is None:
            raise ValueError('no decoder')
        if self._telling:
            dec_buffer, dec_flags = self._decoder.getstate()
        if self._has_read1:
            input_chunk = self.buffer.read1(self._CHUNK_SIZE)
        else:
            input_chunk = self.buffer.read(self._CHUNK_SIZE)
        eof = not input_chunk
        decoded_chars = self._decoder.decode(input_chunk, eof)
        self._set_decoded_chars(decoded_chars)
        if decoded_chars:
            self._b2cratio = len(input_chunk) / len(self._decoded_chars)
        else:
            self._b2cratio = 0.0
        if self._telling:
            self._snapshot = (dec_flags, dec_buffer + input_chunk)
        return not eof

    def _pack_cookie(self, position, dec_flags=0, bytes_to_feed=0, need_eof=False, chars_to_skip=0):
        return position | dec_flags << 64 | bytes_to_feed << 128 | chars_to_skip << 192 | bool(need_eof) << 256

    def _unpack_cookie(self, bigint):
        rest, position = divmod(bigint, 1 << 64)
        rest, dec_flags = divmod(rest, 1 << 64)
        rest, bytes_to_feed = divmod(rest, 1 << 64)
        need_eof, chars_to_skip = divmod(rest, 1 << 64)
        return (position, dec_flags, bytes_to_feed, bool(need_eof), chars_to_skip)

    def tell(self):
        if not self._seekable:
            raise UnsupportedOperation('underlying stream is not seekable')
        if not self._telling:
            raise OSError('telling position disabled by next() call')
        self.flush()
        position = self.buffer.tell()
        decoder = self._decoder
        if decoder is None or self._snapshot is None:
            if self._decoded_chars:
                raise AssertionError('pending decoded text')
            return position
        dec_flags, next_input = self._snapshot
        position -= len(next_input)
        chars_to_skip = self._decoded_chars_used
        if chars_to_skip == 0:
            return self._pack_cookie(position, dec_flags)
        saved_state = decoder.getstate()
        try:
            skip_bytes = int(self._b2cratio * chars_to_skip)
            skip_back = 1
            assert skip_bytes <= len(next_input)
            while skip_bytes > 0:
                decoder.setstate((b'', dec_flags))
                n = len(decoder.decode(next_input[:skip_bytes]))
                if n <= chars_to_skip:
                    b, d = decoder.getstate()
                    if not b:
                        dec_flags = d
                        chars_to_skip -= n
                        break
                    skip_bytes -= len(b)
                    skip_back = 1
                else:
                    skip_bytes -= skip_back
                    skip_back = skip_back * 2
            else:
                skip_bytes = 0
                decoder.setstate((b'', dec_flags))
            start_pos = position + skip_bytes
            start_flags = dec_flags
            if chars_to_skip == 0:
                return self._pack_cookie(start_pos, start_flags)
            bytes_fed = 0
            need_eof = False
            chars_decoded = 0
            for i in range(skip_bytes, len(next_input)):
                bytes_fed += 1
                chars_decoded += len(decoder.decode(next_input[i:i + 1]))
                dec_buffer, dec_flags = decoder.getstate()
                if not dec_buffer and chars_decoded <= chars_to_skip:
                    start_pos += bytes_fed
                    chars_to_skip -= chars_decoded
                    start_flags, bytes_fed, chars_decoded = (dec_flags, 0, 0)
                if chars_decoded >= chars_to_skip:
                    break
            else:
                chars_decoded += len(decoder.decode(b'', final=True))
                need_eof = True
                if chars_decoded < chars_to_skip:
                    raise OSError("can't reconstruct logical file position")
            return self._pack_cookie(start_pos, start_flags, bytes_fed, need_eof, chars_to_skip)
        finally:
            decoder.setstate(saved_state)

    def truncate(self, pos=None):
        self.flush()
        if pos is None:
            pos = self.tell()
        return self.buffer.truncate(pos)

    def detach(self):
        if self.buffer is None:
            raise ValueError('buffer is already detached')
        self.flush()
        buffer = self._buffer
        self._buffer = None
        return buffer

    def seek(self, cookie, whence=0):

        def _reset_encoder(position):
            """Reset the encoder (merely useful for proper BOM handling)"""
            try:
                encoder = self._encoder or self._get_encoder()
            except LookupError:
                pass
            else:
                if position != 0:
                    encoder.setstate(0)
                else:
                    encoder.reset()
        if self.closed:
            raise ValueError('tell on closed file')
        if not self._seekable:
            raise UnsupportedOperation('underlying stream is not seekable')
        if whence == SEEK_CUR:
            if cookie != 0:
                raise UnsupportedOperation("can't do nonzero cur-relative seeks")
            whence = 0
            cookie = self.tell()
        elif whence == SEEK_END:
            if cookie != 0:
                raise UnsupportedOperation("can't do nonzero end-relative seeks")
            self.flush()
            position = self.buffer.seek(0, whence)
            self._set_decoded_chars('')
            self._snapshot = None
            if self._decoder:
                self._decoder.reset()
            _reset_encoder(position)
            return position
        if whence != 0:
            raise ValueError('unsupported whence (%r)' % (whence,))
        if cookie < 0:
            raise ValueError('negative seek position %r' % (cookie,))
        self.flush()
        start_pos, dec_flags, bytes_to_feed, need_eof, chars_to_skip = self._unpack_cookie(cookie)
        self.buffer.seek(start_pos)
        self._set_decoded_chars('')
        self._snapshot = None
        if cookie == 0 and self._decoder:
            self._decoder.reset()
        elif self._decoder or dec_flags or chars_to_skip:
            self._decoder = self._decoder or self._get_decoder()
            self._decoder.setstate((b'', dec_flags))
            self._snapshot = (dec_flags, b'')
        if chars_to_skip:
            input_chunk = self.buffer.read(bytes_to_feed)
            self._set_decoded_chars(self._decoder.decode(input_chunk, need_eof))
            self._snapshot = (dec_flags, input_chunk)
            if len(self._decoded_chars) < chars_to_skip:
                raise OSError("can't restore logical file position")
            self._decoded_chars_used = chars_to_skip
        _reset_encoder(cookie)
        return cookie

    def read(self, size=None):
        self._checkReadable()
        if size is None:
            size = -1
        else:
            try:
                size_index = size.__index__
            except AttributeError:
                raise TypeError(f'{size!r} is not an integer')
            else:
                size = size_index()
        decoder = self._decoder or self._get_decoder()
        if size < 0:
            result = self._get_decoded_chars() + decoder.decode(self.buffer.read(), final=True)
            if self._snapshot is not None:
                self._set_decoded_chars('')
                self._snapshot = None
            return result
        else:
            eof = False
            result = self._get_decoded_chars(size)
            while len(result) < size and (not eof):
                eof = not self._read_chunk()
                result += self._get_decoded_chars(size - len(result))
            return result

    def __next__(self):
        self._telling = False
        line = self.readline()
        if not line:
            self._snapshot = None
            self._telling = self._seekable
            raise StopIteration
        return line

    def readline(self, size=None):
        if self.closed:
            raise ValueError('read from closed file')
        if size is None:
            size = -1
        else:
            try:
                size_index = size.__index__
            except AttributeError:
                raise TypeError(f'{size!r} is not an integer')
            else:
                size = size_index()
        line = self._get_decoded_chars()
        start = 0
        if not self._decoder:
            self._get_decoder()
        pos = endpos = None
        while True:
            if self._readtranslate:
                pos = line.find('\n', start)
                if pos >= 0:
                    endpos = pos + 1
                    break
                else:
                    start = len(line)
            elif self._readuniversal:
                nlpos = line.find('\n', start)
                crpos = line.find('\r', start)
                if crpos == -1:
                    if nlpos == -1:
                        start = len(line)
                    else:
                        endpos = nlpos + 1
                        break
                elif nlpos == -1:
                    endpos = crpos + 1
                    break
                elif nlpos < crpos:
                    endpos = nlpos + 1
                    break
                elif nlpos == crpos + 1:
                    endpos = crpos + 2
                    break
                else:
                    endpos = crpos + 1
                    break
            else:
                pos = line.find(self._readnl)
                if pos >= 0:
                    endpos = pos + len(self._readnl)
                    break
            if size >= 0 and len(line) >= size:
                endpos = size
                break
            while self._read_chunk():
                if self._decoded_chars:
                    break
            if self._decoded_chars:
                line += self._get_decoded_chars()
            else:
                self._set_decoded_chars('')
                self._snapshot = None
                return line
        if size >= 0 and endpos > size:
            endpos = size
        self._rewind_decoded_chars(len(line) - endpos)
        return line[:endpos]

    @property
    def newlines(self):
        return self._decoder.newlines if self._decoder else None