import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
class NetstringReceiver(protocol.Protocol):
    """
    A protocol that sends and receives netstrings.

    See U{http://cr.yp.to/proto/netstrings.txt} for the specification of
    netstrings. Every netstring starts with digits that specify the length
    of the data. This length specification is separated from the data by
    a colon. The data is terminated with a comma.

    Override L{stringReceived} to handle received netstrings. This
    method is called with the netstring payload as a single argument
    whenever a complete netstring is received.

    Security features:
        1. Messages are limited in size, useful if you don't want
           someone sending you a 500MB netstring (change C{self.MAX_LENGTH}
           to the maximum length you wish to accept).
        2. The connection is lost if an illegal message is received.

    @ivar MAX_LENGTH: Defines the maximum length of netstrings that can be
        received.
    @type MAX_LENGTH: C{int}

    @ivar _LENGTH: A pattern describing all strings that contain a netstring
        length specification. Examples for length specifications are C{b'0:'},
        C{b'12:'}, and C{b'179:'}. C{b'007:'} is not a valid length
        specification, since leading zeros are not allowed.
    @type _LENGTH: C{re.Match}

    @ivar _LENGTH_PREFIX: A pattern describing all strings that contain
        the first part of a netstring length specification (without the
        trailing comma). Examples are '0', '12', and '179'. '007' does not
        start a netstring length specification, since leading zeros are
        not allowed.
    @type _LENGTH_PREFIX: C{re.Match}

    @ivar _PARSING_LENGTH: Indicates that the C{NetstringReceiver} is in
        the state of parsing the length portion of a netstring.
    @type _PARSING_LENGTH: C{int}

    @ivar _PARSING_PAYLOAD: Indicates that the C{NetstringReceiver} is in
        the state of parsing the payload portion (data and trailing comma)
        of a netstring.
    @type _PARSING_PAYLOAD: C{int}

    @ivar brokenPeer: Indicates if the connection is still functional
    @type brokenPeer: C{int}

    @ivar _state: Indicates if the protocol is consuming the length portion
        (C{PARSING_LENGTH}) or the payload (C{PARSING_PAYLOAD}) of a netstring
    @type _state: C{int}

    @ivar _remainingData: Holds the chunk of data that has not yet been consumed
    @type _remainingData: C{string}

    @ivar _payload: Holds the payload portion of a netstring including the
        trailing comma
    @type _payload: C{BytesIO}

    @ivar _expectedPayloadSize: Holds the payload size plus one for the trailing
        comma.
    @type _expectedPayloadSize: C{int}
    """
    MAX_LENGTH = 99999
    _LENGTH = re.compile(b'(0|[1-9]\\d*)(:)')
    _LENGTH_PREFIX = re.compile(b'(0|[1-9]\\d*)$')
    _MISSING_LENGTH = 'The received netstring does not start with a length specification.'
    _OVERFLOW = 'The length specification of the received netstring cannot be represented in Python - it causes an OverflowError!'
    _TOO_LONG = 'The received netstring is longer than the maximum %s specified by self.MAX_LENGTH'
    _MISSING_COMMA = 'The received netstring is not terminated by a comma.'
    _PARSING_LENGTH, _PARSING_PAYLOAD = range(2)

    def makeConnection(self, transport):
        """
        Initializes the protocol.
        """
        protocol.Protocol.makeConnection(self, transport)
        self._remainingData = b''
        self._currentPayloadSize = 0
        self._payload = BytesIO()
        self._state = self._PARSING_LENGTH
        self._expectedPayloadSize = 0
        self.brokenPeer = 0

    def sendString(self, string):
        """
        Sends a netstring.

        Wraps up C{string} by adding length information and a
        trailing comma; writes the result to the transport.

        @param string: The string to send.  The necessary framing (length
            prefix, etc) will be added.
        @type string: C{bytes}
        """
        self.transport.write(_formatNetstring(string))

    def dataReceived(self, data):
        """
        Receives some characters of a netstring.

        Whenever a complete netstring is received, this method extracts
        its payload and calls L{stringReceived} to process it.

        @param data: A chunk of data representing a (possibly partial)
            netstring
        @type data: C{bytes}
        """
        self._remainingData += data
        while self._remainingData:
            try:
                self._consumeData()
            except IncompleteNetstring:
                break
            except NetstringParseError:
                self._handleParseError()
                break

    def stringReceived(self, string):
        """
        Override this for notification when each complete string is received.

        @param string: The complete string which was received with all
            framing (length prefix, etc) removed.
        @type string: C{bytes}

        @raise NotImplementedError: because the method has to be implemented
            by the child class.
        """
        raise NotImplementedError()

    def _maxLengthSize(self):
        """
        Calculate and return the string size of C{self.MAX_LENGTH}.

        @return: The size of the string representation for C{self.MAX_LENGTH}
        @rtype: C{float}
        """
        return math.ceil(math.log10(self.MAX_LENGTH)) + 1

    def _consumeData(self):
        """
        Consumes the content of C{self._remainingData}.

        @raise IncompleteNetstring: if C{self._remainingData} does not
            contain enough data to complete the current netstring.
        @raise NetstringParseError: if the received data do not
            form a valid netstring.
        """
        if self._state == self._PARSING_LENGTH:
            self._consumeLength()
            self._prepareForPayloadConsumption()
        if self._state == self._PARSING_PAYLOAD:
            self._consumePayload()

    def _consumeLength(self):
        """
        Consumes the length portion of C{self._remainingData}.

        @raise IncompleteNetstring: if C{self._remainingData} contains
            a partial length specification (digits without trailing
            comma).
        @raise NetstringParseError: if the received data do not form a valid
            netstring.
        """
        lengthMatch = self._LENGTH.match(self._remainingData)
        if not lengthMatch:
            self._checkPartialLengthSpecification()
            raise IncompleteNetstring()
        self._processLength(lengthMatch)

    def _checkPartialLengthSpecification(self):
        """
        Makes sure that the received data represents a valid number.

        Checks if C{self._remainingData} represents a number smaller or
        equal to C{self.MAX_LENGTH}.

        @raise NetstringParseError: if C{self._remainingData} is no
            number or is too big (checked by L{_extractLength}).
        """
        partialLengthMatch = self._LENGTH_PREFIX.match(self._remainingData)
        if not partialLengthMatch:
            raise NetstringParseError(self._MISSING_LENGTH)
        lengthSpecification = partialLengthMatch.group(1)
        self._extractLength(lengthSpecification)

    def _processLength(self, lengthMatch):
        """
        Processes the length definition of a netstring.

        Extracts and stores in C{self._expectedPayloadSize} the number
        representing the netstring size.  Removes the prefix
        representing the length specification from
        C{self._remainingData}.

        @raise NetstringParseError: if the received netstring does not
            start with a number or the number is bigger than
            C{self.MAX_LENGTH}.
        @param lengthMatch: A regular expression match object matching
            a netstring length specification
        @type lengthMatch: C{re.Match}
        """
        endOfNumber = lengthMatch.end(1)
        startOfData = lengthMatch.end(2)
        lengthString = self._remainingData[:endOfNumber]
        self._expectedPayloadSize = self._extractLength(lengthString) + 1
        self._remainingData = self._remainingData[startOfData:]

    def _extractLength(self, lengthAsString):
        """
        Attempts to extract the length information of a netstring.

        @raise NetstringParseError: if the number is bigger than
            C{self.MAX_LENGTH}.
        @param lengthAsString: A chunk of data starting with a length
            specification
        @type lengthAsString: C{bytes}
        @return: The length of the netstring
        @rtype: C{int}
        """
        self._checkStringSize(lengthAsString)
        length = int(lengthAsString)
        if length > self.MAX_LENGTH:
            raise NetstringParseError(self._TOO_LONG % (self.MAX_LENGTH,))
        return length

    def _checkStringSize(self, lengthAsString):
        """
        Checks the sanity of lengthAsString.

        Checks if the size of the length specification exceeds the
        size of the string representing self.MAX_LENGTH. If this is
        not the case, the number represented by lengthAsString is
        certainly bigger than self.MAX_LENGTH, and a
        NetstringParseError can be raised.

        This method should make sure that netstrings with extremely
        long length specifications are refused before even attempting
        to convert them to an integer (which might trigger a
        MemoryError).
        """
        if len(lengthAsString) > self._maxLengthSize():
            raise NetstringParseError(self._TOO_LONG % (self.MAX_LENGTH,))

    def _prepareForPayloadConsumption(self):
        """
        Sets up variables necessary for consuming the payload of a netstring.
        """
        self._state = self._PARSING_PAYLOAD
        self._currentPayloadSize = 0
        self._payload.seek(0)
        self._payload.truncate()

    def _consumePayload(self):
        """
        Consumes the payload portion of C{self._remainingData}.

        If the payload is complete, checks for the trailing comma and
        processes the payload. If not, raises an L{IncompleteNetstring}
        exception.

        @raise IncompleteNetstring: if the payload received so far
            contains fewer characters than expected.
        @raise NetstringParseError: if the payload does not end with a
        comma.
        """
        self._extractPayload()
        if self._currentPayloadSize < self._expectedPayloadSize:
            raise IncompleteNetstring()
        self._checkForTrailingComma()
        self._state = self._PARSING_LENGTH
        self._processPayload()

    def _extractPayload(self):
        """
        Extracts payload information from C{self._remainingData}.

        Splits C{self._remainingData} at the end of the netstring.  The
        first part becomes C{self._payload}, the second part is stored
        in C{self._remainingData}.

        If the netstring is not yet complete, the whole content of
        C{self._remainingData} is moved to C{self._payload}.
        """
        if self._payloadComplete():
            remainingPayloadSize = self._expectedPayloadSize - self._currentPayloadSize
            self._payload.write(self._remainingData[:remainingPayloadSize])
            self._remainingData = self._remainingData[remainingPayloadSize:]
            self._currentPayloadSize = self._expectedPayloadSize
        else:
            self._payload.write(self._remainingData)
            self._currentPayloadSize += len(self._remainingData)
            self._remainingData = b''

    def _payloadComplete(self):
        """
        Checks if enough data have been received to complete the netstring.

        @return: C{True} iff the received data contain at least as many
            characters as specified in the length section of the
            netstring
        @rtype: C{bool}
        """
        return len(self._remainingData) + self._currentPayloadSize >= self._expectedPayloadSize

    def _processPayload(self):
        """
        Processes the actual payload with L{stringReceived}.

        Strips C{self._payload} of the trailing comma and calls
        L{stringReceived} with the result.
        """
        self.stringReceived(self._payload.getvalue()[:-1])

    def _checkForTrailingComma(self):
        """
        Checks if the netstring has a trailing comma at the expected position.

        @raise NetstringParseError: if the last payload character is
            anything but a comma.
        """
        if self._payload.getvalue()[-1:] != b',':
            raise NetstringParseError(self._MISSING_COMMA)

    def _handleParseError(self):
        """
        Terminates the connection and sets the flag C{self.brokenPeer}.
        """
        self.transport.loseConnection()
        self.brokenPeer = 1