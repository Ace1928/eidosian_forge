import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock
class ABNF(object):
    """
    ABNF frame class.
    see http://tools.ietf.org/html/rfc5234
    and http://tools.ietf.org/html/rfc6455#section-5.2
    """
    OPCODE_CONT = 0
    OPCODE_TEXT = 1
    OPCODE_BINARY = 2
    OPCODE_CLOSE = 8
    OPCODE_PING = 9
    OPCODE_PONG = 10
    OPCODES = (OPCODE_CONT, OPCODE_TEXT, OPCODE_BINARY, OPCODE_CLOSE, OPCODE_PING, OPCODE_PONG)
    OPCODE_MAP = {OPCODE_CONT: 'cont', OPCODE_TEXT: 'text', OPCODE_BINARY: 'binary', OPCODE_CLOSE: 'close', OPCODE_PING: 'ping', OPCODE_PONG: 'pong'}
    LENGTH_7 = 126
    LENGTH_16 = 1 << 16
    LENGTH_63 = 1 << 63

    def __init__(self, fin=0, rsv1=0, rsv2=0, rsv3=0, opcode=OPCODE_TEXT, mask=1, data=''):
        """
        Constructor for ABNF.
        please check RFC for arguments.
        """
        self.fin = fin
        self.rsv1 = rsv1
        self.rsv2 = rsv2
        self.rsv3 = rsv3
        self.opcode = opcode
        self.mask = mask
        if data is None:
            data = ''
        self.data = data
        self.get_mask_key = os.urandom

    def validate(self, skip_utf8_validation=False):
        """
        validate the ABNF frame.
        skip_utf8_validation: skip utf8 validation.
        """
        if self.rsv1 or self.rsv2 or self.rsv3:
            raise WebSocketProtocolException('rsv is not implemented, yet')
        if self.opcode not in ABNF.OPCODES:
            raise WebSocketProtocolException('Invalid opcode %r', self.opcode)
        if self.opcode == ABNF.OPCODE_PING and (not self.fin):
            raise WebSocketProtocolException('Invalid ping frame.')
        if self.opcode == ABNF.OPCODE_CLOSE:
            l = len(self.data)
            if not l:
                return
            if l == 1 or l >= 126:
                raise WebSocketProtocolException('Invalid close frame.')
            if l > 2 and (not skip_utf8_validation) and (not validate_utf8(self.data[2:])):
                raise WebSocketProtocolException('Invalid close frame.')
            code = 256 * six.byte2int(self.data[0:1]) + six.byte2int(self.data[1:2])
            if not self._is_valid_close_status(code):
                raise WebSocketProtocolException('Invalid close opcode.')

    @staticmethod
    def _is_valid_close_status(code):
        return code in VALID_CLOSE_STATUS or 3000 <= code < 5000

    def __str__(self):
        return 'fin=' + str(self.fin) + ' opcode=' + str(self.opcode) + ' data=' + str(self.data)

    @staticmethod
    def create_frame(data, opcode, fin=1):
        """
        create frame to send text, binary and other data.

        data: data to send. This is string value(byte array).
            if opcode is OPCODE_TEXT and this value is unicode,
            data value is converted into unicode string, automatically.

        opcode: operation code. please see OPCODE_XXX.

        fin: fin flag. if set to 0, create continue fragmentation.
        """
        if opcode == ABNF.OPCODE_TEXT and isinstance(data, six.text_type):
            data = data.encode('utf-8')
        return ABNF(fin, 0, 0, 0, opcode, 1, data)

    def format(self):
        """
        format this object to string(byte array) to send data to server.
        """
        if any((x not in (0, 1) for x in [self.fin, self.rsv1, self.rsv2, self.rsv3])):
            raise ValueError('not 0 or 1')
        if self.opcode not in ABNF.OPCODES:
            raise ValueError('Invalid OPCODE')
        length = len(self.data)
        if length >= ABNF.LENGTH_63:
            raise ValueError('data is too long')
        frame_header = chr(self.fin << 7 | self.rsv1 << 6 | self.rsv2 << 5 | self.rsv3 << 4 | self.opcode)
        if length < ABNF.LENGTH_7:
            frame_header += chr(self.mask << 7 | length)
            frame_header = six.b(frame_header)
        elif length < ABNF.LENGTH_16:
            frame_header += chr(self.mask << 7 | 126)
            frame_header = six.b(frame_header)
            frame_header += struct.pack('!H', length)
        else:
            frame_header += chr(self.mask << 7 | 127)
            frame_header = six.b(frame_header)
            frame_header += struct.pack('!Q', length)
        if not self.mask:
            return frame_header + self.data
        else:
            mask_key = self.get_mask_key(4)
            return frame_header + self._get_masked(mask_key)

    def _get_masked(self, mask_key):
        s = ABNF.mask(mask_key, self.data)
        if isinstance(mask_key, six.text_type):
            mask_key = mask_key.encode('utf-8')
        return mask_key + s

    @staticmethod
    def mask(mask_key, data):
        """
        mask or unmask data. Just do xor for each byte

        mask_key: 4 byte string(byte).

        data: data to mask/unmask.
        """
        if data is None:
            data = ''
        if isinstance(mask_key, six.text_type):
            mask_key = six.b(mask_key)
        if isinstance(data, six.text_type):
            data = six.b(data)
        if numpy:
            origlen = len(data)
            _mask_key = mask_key[3] << 24 | mask_key[2] << 16 | mask_key[1] << 8 | mask_key[0]
            data += bytes(' ' * (4 - len(data) % 4), 'us-ascii')
            a = numpy.frombuffer(data, dtype='uint32')
            masked = numpy.bitwise_xor(a, [_mask_key]).astype('uint32')
            if len(data) > origlen:
                return masked.tobytes()[:origlen]
            return masked.tobytes()
        else:
            _m = array.array('B', mask_key)
            _d = array.array('B', data)
            return _mask(_m, _d)