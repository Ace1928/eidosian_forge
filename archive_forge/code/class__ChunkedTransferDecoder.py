from __future__ import annotations
import base64
import binascii
import calendar
import math
import os
import re
import tempfile
import time
import warnings
from email import message_from_bytes
from email.message import EmailMessage
from io import BytesIO
from typing import AnyStr, Callable, Dict, List, Optional, Tuple
from urllib.parse import (
from zope.interface import Attribute, Interface, implementer, provider
from incremental import Version
from twisted.internet import address, interfaces, protocol
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IProtocol
from twisted.logger import Logger
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString, networkString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.web._responses import (
from twisted.web.http_headers import Headers, _sanitizeLinearWhitespace
from twisted.web.iweb import IAccessLogFormatter, INonQueuedRequestFactory, IRequest
class _ChunkedTransferDecoder:
    """
    Protocol for decoding I{chunked} Transfer-Encoding, as defined by RFC 7230,
    section 4.1.  This protocol can interpret the contents of a request or
    response body which uses the I{chunked} Transfer-Encoding.  It cannot
    interpret any of the rest of the HTTP protocol.

    It may make sense for _ChunkedTransferDecoder to be an actual IProtocol
    implementation.  Currently, the only user of this class will only ever
    call dataReceived on it.  However, it might be an improvement if the
    user could connect this to a transport and deliver connection lost
    notification.  This way, `dataCallback` becomes `self.transport.write`
    and perhaps `finishCallback` becomes `self.transport.loseConnection()`
    (although I'm not sure where the extra data goes in that case).  This
    could also allow this object to indicate to the receiver of data that
    the stream was not completely received, an error case which should be
    noticed. -exarkun

    @ivar dataCallback: A one-argument callable which will be invoked each
        time application data is received. This callback is not reentrant.

    @ivar finishCallback: A one-argument callable which will be invoked when
        the terminal chunk is received.  It will be invoked with all bytes
        which were delivered to this protocol which came after the terminal
        chunk.

    @ivar length: Counter keeping track of how many more bytes in a chunk there
        are to receive.

    @ivar state: One of C{'CHUNK_LENGTH'}, C{'CRLF'}, C{'TRAILER'},
        C{'BODY'}, or C{'FINISHED'}.  For C{'CHUNK_LENGTH'}, data for the
        chunk length line is currently being read.  For C{'CRLF'}, the CR LF
        pair which follows each chunk is being read. For C{'TRAILER'}, the CR
        LF pair which follows the terminal 0-length chunk is currently being
        read. For C{'BODY'}, the contents of a chunk are being read. For
        C{'FINISHED'}, the last chunk has been completely read and no more
        input is valid.

    @ivar _buffer: Accumulated received data for the current state. At each
        state transition this is truncated at the front so that index 0 is
        where the next state shall begin.

    @ivar _start: While in the C{'CHUNK_LENGTH'} and C{'TRAILER'} states,
        tracks the index into the buffer at which search for CRLF should resume.
        Resuming the search at this position avoids doing quadratic work if the
        chunk length line arrives over many calls to C{dataReceived}.

    @ivar _trailerHeaders: Accumulates raw/unparsed trailer headers.
        See https://github.com/twisted/twisted/issues/12014

    @ivar _maxTrailerHeadersSize: Maximum bytes for trailer header from the
        response.
    @type _maxTrailerHeadersSize: C{int}

    @ivar _receivedTrailerHeadersSize: Bytes received so far for the tailer headers.
    @type _receivedTrailerHeadersSize: C{int}
    """
    state = 'CHUNK_LENGTH'

    def __init__(self, dataCallback: Callable[[bytes], None], finishCallback: Callable[[bytes], None]) -> None:
        self.dataCallback = dataCallback
        self.finishCallback = finishCallback
        self._buffer = bytearray()
        self._start = 0
        self._trailerHeaders: List[bytearray] = []
        self._maxTrailerHeadersSize = 2 ** 16
        self._receivedTrailerHeadersSize = 0

    def _dataReceived_CHUNK_LENGTH(self) -> bool:
        """
        Read the chunk size line, ignoring any extensions.

        @returns: C{True} once the line has been read and removed from
            C{self._buffer}.  C{False} when more data is required.

        @raises _MalformedChunkedDataError: when the chunk size cannot be
            decoded or the length of the line exceeds L{maxChunkSizeLineLength}.
        """
        eolIndex = self._buffer.find(b'\r\n', self._start)
        if eolIndex >= maxChunkSizeLineLength or (eolIndex == -1 and len(self._buffer) > maxChunkSizeLineLength):
            raise _MalformedChunkedDataError('Chunk size line exceeds maximum of {} bytes.'.format(maxChunkSizeLineLength))
        if eolIndex == -1:
            self._start = len(self._buffer) - 1
            return False
        endOfLengthIndex = self._buffer.find(b';', 0, eolIndex)
        if endOfLengthIndex == -1:
            endOfLengthIndex = eolIndex
        rawLength = self._buffer[0:endOfLengthIndex]
        try:
            length = _hexint(rawLength)
        except ValueError:
            raise _MalformedChunkedDataError('Chunk-size must be an integer.')
        ext = self._buffer[endOfLengthIndex + 1:eolIndex]
        if ext and ext.translate(None, _chunkExtChars) != b'':
            raise _MalformedChunkedDataError(f'Invalid characters in chunk extensions: {ext!r}.')
        if length == 0:
            self.state = 'TRAILER'
        else:
            self.state = 'BODY'
        self.length = length
        del self._buffer[0:eolIndex + 2]
        self._start = 0
        return True

    def _dataReceived_CRLF(self) -> bool:
        """
        Await the carriage return and line feed characters that are the end of
        chunk marker that follow the chunk data.

        @returns: C{True} when the CRLF have been read, otherwise C{False}.

        @raises _MalformedChunkedDataError: when anything other than CRLF are
            received.
        """
        if len(self._buffer) < 2:
            return False
        if not self._buffer.startswith(b'\r\n'):
            raise _MalformedChunkedDataError('Chunk did not end with CRLF')
        self.state = 'CHUNK_LENGTH'
        del self._buffer[0:2]
        return True

    def _dataReceived_TRAILER(self) -> bool:
        """
        Collect trailer headers if received and finish at the terminal zero-length
        chunk. Then invoke C{finishCallback} and switch to state C{'FINISHED'}.

        @returns: C{False}, as there is either insufficient data to continue,
            or no data remains.
        """
        if self._receivedTrailerHeadersSize + len(self._buffer) > self._maxTrailerHeadersSize:
            raise _MalformedChunkedDataError('Trailer headers data is too long.')
        eolIndex = self._buffer.find(b'\r\n', self._start)
        if eolIndex == -1:
            return False
        if eolIndex > 0:
            self._trailerHeaders.append(self._buffer[0:eolIndex])
            del self._buffer[0:eolIndex + 2]
            self._start = 0
            self._receivedTrailerHeadersSize += eolIndex + 2
            return True
        data = memoryview(self._buffer)[2:].tobytes()
        del self._buffer[:]
        self.state = 'FINISHED'
        self.finishCallback(data)
        return False

    def _dataReceived_BODY(self) -> bool:
        """
        Deliver any available chunk data to the C{dataCallback}. When all the
        remaining data for the chunk arrives, switch to state C{'CRLF'}.

        @returns: C{True} to continue processing of any buffered data.
        """
        if len(self._buffer) >= self.length:
            chunk = memoryview(self._buffer)[:self.length].tobytes()
            del self._buffer[:self.length]
            self.state = 'CRLF'
            self.dataCallback(chunk)
        else:
            chunk = bytes(self._buffer)
            self.length -= len(chunk)
            del self._buffer[:]
            self.dataCallback(chunk)
        return True

    def _dataReceived_FINISHED(self) -> bool:
        """
        Once C{finishCallback} has been invoked receipt of additional data
        raises L{RuntimeError} because it represents a programming error in
        the caller.
        """
        raise RuntimeError('_ChunkedTransferDecoder.dataReceived called after last chunk was processed')

    def dataReceived(self, data: bytes) -> None:
        """
        Interpret data from a request or response body which uses the
        I{chunked} Transfer-Encoding.
        """
        self._buffer += data
        goOn = True
        while goOn and self._buffer:
            goOn = getattr(self, '_dataReceived_' + self.state)()

    def noMoreData(self) -> None:
        """
        Verify that all data has been received.  If it has not been, raise
        L{_DataLoss}.
        """
        if self.state != 'FINISHED':
            raise _DataLoss("Chunked decoder in %r state, still expecting more data to get to 'FINISHED' state." % (self.state,))