import binascii
import struct
from . import packet_base
from os_ken.lib import addrconv
class bpdu(packet_base.PacketBase):
    """Bridge Protocol Data Unit(BPDU) header encoder/decoder base class.
    """
    _PACK_STR = '!HBB'
    _PACK_LEN = struct.calcsize(_PACK_STR)
    _BPDU_TYPES = {}
    _MIN_LEN = _PACK_LEN

    @staticmethod
    def register_bpdu_type(sub_cls):
        bpdu._BPDU_TYPES.setdefault(sub_cls.VERSION_ID, {})
        bpdu._BPDU_TYPES[sub_cls.VERSION_ID][sub_cls.BPDU_TYPE] = sub_cls
        return sub_cls

    def __init__(self):
        super(bpdu, self).__init__()
        assert hasattr(self, 'VERSION_ID')
        assert hasattr(self, 'BPDU_TYPE')
        self._protocol_id = PROTOCOL_IDENTIFIER
        self._version_id = self.VERSION_ID
        self._bpdu_type = self.BPDU_TYPE
        if hasattr(self, 'check_parameters'):
            self.check_parameters()

    @classmethod
    def parser(cls, buf):
        assert len(buf) >= cls._PACK_LEN
        protocol_id, version_id, bpdu_type = struct.unpack_from(cls._PACK_STR, buf)
        assert protocol_id == PROTOCOL_IDENTIFIER
        if version_id in cls._BPDU_TYPES and bpdu_type in cls._BPDU_TYPES[version_id]:
            bpdu_cls = cls._BPDU_TYPES[version_id][bpdu_type]
            assert len(buf[cls._PACK_LEN:]) >= bpdu_cls.PACK_LEN
            return bpdu_cls.parser(buf[cls._PACK_LEN:])
        else:
            return (buf, None, None)

    def serialize(self, payload, prev):
        return struct.pack(bpdu._PACK_STR, self._protocol_id, self._version_id, self._bpdu_type)