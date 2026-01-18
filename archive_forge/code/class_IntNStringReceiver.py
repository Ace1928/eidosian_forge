import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
class IntNStringReceiver(protocol.Protocol, _PauseableMixin):
    """
    Generic class for length prefixed protocols.

    @ivar _unprocessed: bytes received, but not yet broken up into messages /
        sent to stringReceived.  _compatibilityOffset must be updated when this
        value is updated so that the C{recvd} attribute can be generated
        correctly.
    @type _unprocessed: C{bytes}

    @ivar structFormat: format used for struct packing/unpacking. Define it in
        subclass.
    @type structFormat: C{str}

    @ivar prefixLength: length of the prefix, in bytes. Define it in subclass,
        using C{struct.calcsize(structFormat)}
    @type prefixLength: C{int}

    @ivar _compatibilityOffset: the offset within C{_unprocessed} to the next
        message to be parsed. (used to generate the recvd attribute)
    @type _compatibilityOffset: C{int}
    """
    MAX_LENGTH = 99999
    _unprocessed = b''
    _compatibilityOffset = 0
    recvd = _RecvdCompatHack()

    def stringReceived(self, string):
        """
        Override this for notification when each complete string is received.

        @param string: The complete string which was received with all
            framing (length prefix, etc) removed.
        @type string: C{bytes}
        """
        raise NotImplementedError

    def lengthLimitExceeded(self, length):
        """
        Callback invoked when a length prefix greater than C{MAX_LENGTH} is
        received.  The default implementation disconnects the transport.
        Override this.

        @param length: The length prefix which was received.
        @type length: C{int}
        """
        self.transport.loseConnection()

    def dataReceived(self, data):
        """
        Convert int prefixed strings into calls to stringReceived.
        """
        alldata = self._unprocessed + data
        currentOffset = 0
        prefixLength = self.prefixLength
        fmt = self.structFormat
        self._unprocessed = alldata
        while len(alldata) >= currentOffset + prefixLength and (not self.paused):
            messageStart = currentOffset + prefixLength
            length, = unpack(fmt, alldata[currentOffset:messageStart])
            if length > self.MAX_LENGTH:
                self._unprocessed = alldata
                self._compatibilityOffset = currentOffset
                self.lengthLimitExceeded(length)
                return
            messageEnd = messageStart + length
            if len(alldata) < messageEnd:
                break
            packet = alldata[messageStart:messageEnd]
            currentOffset = messageEnd
            self._compatibilityOffset = currentOffset
            self.stringReceived(packet)
            if 'recvd' in self.__dict__:
                alldata = self.__dict__.pop('recvd')
                self._unprocessed = alldata
                self._compatibilityOffset = currentOffset = 0
                if alldata:
                    continue
                return
        self._unprocessed = alldata[currentOffset:]
        self._compatibilityOffset = 0

    def sendString(self, string):
        """
        Send a prefixed string to the other end of the connection.

        @param string: The string to send.  The necessary framing (length
            prefix, etc) will be added.
        @type string: C{bytes}
        """
        if len(string) >= 2 ** (8 * self.prefixLength):
            raise StringTooLongError('Try to send %s bytes whereas maximum is %s' % (len(string), 2 ** (8 * self.prefixLength)))
        self.transport.write(pack(self.structFormat, len(string)) + string)