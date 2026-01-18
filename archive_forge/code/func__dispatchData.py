import errno
from zope.interface import implementer
from twisted.internet import error, interfaces, main
from twisted.internet.abstract import _ConsumerMixin, _dataMustBeBytes, _LogOwner
from twisted.internet.iocpreactor import iocpsupport as _iocp
from twisted.internet.iocpreactor.const import ERROR_HANDLE_EOF, ERROR_IO_PENDING
from twisted.python import failure
def _dispatchData(self):
    """
        Dispatch previously read data. Return True if self.reading and we don't
        have any more data
        """
    if not self._readSize:
        return self.reading
    size = self._readSize
    full_buffers = size // self.readBufferSize
    while self._readNextBuffer < full_buffers:
        self.dataReceived(self._readBuffers[self._readNextBuffer])
        self._readNextBuffer += 1
        if not self.reading:
            return False
    remainder = size % self.readBufferSize
    if remainder:
        self.dataReceived(self._readBuffers[full_buffers][0:remainder])
    if self.dynamicReadBuffers:
        total_buffer_size = self.readBufferSize * len(self._readBuffers)
        if size < total_buffer_size - self.readBufferSize:
            del self._readBuffers[-1]
        elif size == total_buffer_size and len(self._readBuffers) < self.maxReadBuffers:
            self._readBuffers.append(bytearray(self.readBufferSize))
    self._readNextBuffer = 0
    self._readSize = 0
    return self.reading