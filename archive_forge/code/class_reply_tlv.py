import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class reply_tlv(tlv):
    _PACK_STR = '!BHB6s'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _MIN_VALUE_LEN = _MIN_LEN - struct.calcsize('!BH')
    _PORT_ID_LENGTH_LEN = 1
    _PORT_ID_SUBTYPE_LEN = 1

    def __init__(self, length, action, mac_address, port_id_length, port_id_subtype, port_id):
        super(reply_tlv, self).__init__(length)
        self.action = action
        self.mac_address = mac_address
        self.port_id_length = port_id_length
        self.port_id_subtype = port_id_subtype
        self.port_id = port_id

    @classmethod
    def parser(cls, buf):
        type_, length, action, mac_address = struct.unpack_from(cls._PACK_STR, buf)
        port_id_length = 0
        port_id_subtype = 0
        port_id = b''
        if length > cls._MIN_VALUE_LEN:
            port_id_length, port_id_subtype = struct.unpack_from('!2B', buf, cls._MIN_LEN)
            form = '%ds' % port_id_length
            port_id, = struct.unpack_from(form, buf, cls._MIN_LEN + cls._PORT_ID_LENGTH_LEN + cls._PORT_ID_SUBTYPE_LEN)
        return cls(length, action, addrconv.mac.bin_to_text(mac_address), port_id_length, port_id_subtype, port_id)

    def serialize(self):
        if self.port_id_length == 0:
            self.port_id_length = len(self.port_id)
        if self.length == 0:
            self.length = self._MIN_VALUE_LEN
            if self.port_id_length != 0:
                self.length += self.port_id_length + self._PORT_ID_LENGTH_LEN + self._PORT_ID_SUBTYPE_LEN
        buf = struct.pack(self._PACK_STR, self._type, self.length, self.action, addrconv.mac.text_to_bin(self.mac_address))
        buf = bytearray(buf)
        if self.port_id_length != 0:
            buf.extend(struct.pack('!BB', self.port_id_length, self.port_id_subtype))
            form = '%ds' % self.port_id_length
            buf.extend(struct.pack(form, self.port_id))
        return buf