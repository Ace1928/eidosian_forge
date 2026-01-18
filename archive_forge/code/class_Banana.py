import copy
import struct
from io import BytesIO
from twisted.internet import protocol
from twisted.persisted import styles
from twisted.python import log
from twisted.python.compat import iterbytes
from twisted.python.reflect import fullyQualifiedName
class Banana(protocol.Protocol, styles.Ephemeral):
    """
    L{Banana} implements the I{Banana} s-expression protocol, client and
    server.

    @ivar knownDialects: These are the profiles supported by this Banana
        implementation.
    @type knownDialects: L{list} of L{bytes}
    """
    knownDialects = [b'pb', b'none']
    prefixLimit = None
    sizeLimit = SIZE_LIMIT

    def setPrefixLimit(self, limit):
        """
        Set the prefix limit for decoding done by this protocol instance.

        @see: L{setPrefixLimit}
        """
        self.prefixLimit = limit
        self._smallestLongInt = -2 ** (limit * 7) + 1
        self._smallestInt = -2 ** 31
        self._largestInt = 2 ** 31 - 1
        self._largestLongInt = 2 ** (limit * 7) - 1

    def connectionReady(self):
        """Surrogate for connectionMade
        Called after protocol negotiation.
        """

    def _selectDialect(self, dialect):
        self.currentDialect = dialect
        self.connectionReady()

    def callExpressionReceived(self, obj):
        if self.currentDialect:
            self.expressionReceived(obj)
        elif self.isClient:
            for serverVer in obj:
                if serverVer in self.knownDialects:
                    self.sendEncoded(serverVer)
                    self._selectDialect(serverVer)
                    break
            else:
                log.msg("The client doesn't speak any of the protocols offered by the server: disconnecting.")
                self.transport.loseConnection()
        elif obj in self.knownDialects:
            self._selectDialect(obj)
        else:
            log.msg("The client selected a protocol the server didn't suggest and doesn't know: disconnecting.")
            self.transport.loseConnection()

    def connectionMade(self):
        self.setPrefixLimit(_PREFIX_LIMIT)
        self.currentDialect = None
        if not self.isClient:
            self.sendEncoded(self.knownDialects)

    def gotItem(self, item):
        l = self.listStack
        if l:
            l[-1][1].append(item)
        else:
            self.callExpressionReceived(item)
    buffer = b''

    def dataReceived(self, chunk):
        buffer = self.buffer + chunk
        listStack = self.listStack
        gotItem = self.gotItem
        while buffer:
            assert self.buffer != buffer, "This ain't right: {} {}".format(repr(self.buffer), repr(buffer))
            self.buffer = buffer
            pos = 0
            for ch in iterbytes(buffer):
                if ch >= HIGH_BIT_SET:
                    break
                pos = pos + 1
            else:
                if pos > self.prefixLimit:
                    raise BananaError('Security precaution: more than %d bytes of prefix' % (self.prefixLimit,))
                return
            num = buffer[:pos]
            typebyte = buffer[pos:pos + 1]
            rest = buffer[pos + 1:]
            if len(num) > self.prefixLimit:
                raise BananaError('Security precaution: longer than %d bytes worth of prefix' % (self.prefixLimit,))
            if typebyte == LIST:
                num = b1282int(num)
                if num > SIZE_LIMIT:
                    raise BananaError('Security precaution: List too long.')
                listStack.append((num, []))
                buffer = rest
            elif typebyte == STRING:
                num = b1282int(num)
                if num > SIZE_LIMIT:
                    raise BananaError('Security precaution: String too long.')
                if len(rest) >= num:
                    buffer = rest[num:]
                    gotItem(rest[:num])
                else:
                    return
            elif typebyte == INT:
                buffer = rest
                num = b1282int(num)
                gotItem(num)
            elif typebyte == LONGINT:
                buffer = rest
                num = b1282int(num)
                gotItem(num)
            elif typebyte == LONGNEG:
                buffer = rest
                num = b1282int(num)
                gotItem(-num)
            elif typebyte == NEG:
                buffer = rest
                num = -b1282int(num)
                gotItem(num)
            elif typebyte == VOCAB:
                buffer = rest
                num = b1282int(num)
                item = self.incomingVocabulary[num]
                if self.currentDialect == b'pb':
                    gotItem(item)
                else:
                    raise NotImplementedError(f'Invalid item for pb protocol {item!r}')
            elif typebyte == FLOAT:
                if len(rest) >= 8:
                    buffer = rest[8:]
                    gotItem(struct.unpack('!d', rest[:8])[0])
                else:
                    return
            else:
                raise NotImplementedError(f'Invalid Type Byte {typebyte!r}')
            while listStack and len(listStack[-1][1]) == listStack[-1][0]:
                item = listStack.pop()[1]
                gotItem(item)
        self.buffer = b''

    def expressionReceived(self, lst):
        """Called when an expression (list, string, or int) is received."""
        raise NotImplementedError()
    outgoingVocabulary = {b'None': 1, b'class': 2, b'dereference': 3, b'reference': 4, b'dictionary': 5, b'function': 6, b'instance': 7, b'list': 8, b'module': 9, b'persistent': 10, b'tuple': 11, b'unpersistable': 12, b'copy': 13, b'cache': 14, b'cached': 15, b'remote': 16, b'local': 17, b'lcache': 18, b'version': 19, b'login': 20, b'password': 21, b'challenge': 22, b'logged_in': 23, b'not_logged_in': 24, b'cachemessage': 25, b'message': 26, b'answer': 27, b'error': 28, b'decref': 29, b'decache': 30, b'uncache': 31}
    incomingVocabulary = {}
    for k, v in outgoingVocabulary.items():
        incomingVocabulary[v] = k

    def __init__(self, isClient=1):
        self.listStack = []
        self.outgoingSymbols = copy.copy(self.outgoingVocabulary)
        self.outgoingSymbolCount = 0
        self.isClient = isClient

    def sendEncoded(self, obj):
        """
        Send the encoded representation of the given object:

        @param obj: An object to encode and send.

        @raise BananaError: If the given object is not an instance of one of
            the types supported by Banana.

        @return: L{None}
        """
        encodeStream = BytesIO()
        self._encode(obj, encodeStream.write)
        value = encodeStream.getvalue()
        self.transport.write(value)

    def _encode(self, obj, write):
        if isinstance(obj, (list, tuple)):
            if len(obj) > SIZE_LIMIT:
                raise BananaError('list/tuple is too long to send (%d)' % (len(obj),))
            int2b128(len(obj), write)
            write(LIST)
            for elem in obj:
                self._encode(elem, write)
        elif isinstance(obj, int):
            if obj < self._smallestLongInt or obj > self._largestLongInt:
                raise BananaError('int is too large to send (%d)' % (obj,))
            if obj < self._smallestInt:
                int2b128(-obj, write)
                write(LONGNEG)
            elif obj < 0:
                int2b128(-obj, write)
                write(NEG)
            elif obj <= self._largestInt:
                int2b128(obj, write)
                write(INT)
            else:
                int2b128(obj, write)
                write(LONGINT)
        elif isinstance(obj, float):
            write(FLOAT)
            write(struct.pack('!d', obj))
        elif isinstance(obj, bytes):
            if self.currentDialect == b'pb' and obj in self.outgoingSymbols:
                symbolID = self.outgoingSymbols[obj]
                int2b128(symbolID, write)
                write(VOCAB)
            else:
                if len(obj) > SIZE_LIMIT:
                    raise BananaError('byte string is too long to send (%d)' % (len(obj),))
                int2b128(len(obj), write)
                write(STRING)
                write(obj)
        else:
            raise BananaError('Banana cannot send {} objects: {!r}'.format(fullyQualifiedName(type(obj)), obj))