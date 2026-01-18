import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class lldp(packet_base.PacketBase):
    """LLDPDU encoder/decoder class.

    An instance has the following attributes at least.

    ============== =====================================
    Attribute      Description
    ============== =====================================
    tlvs           List of TLV instance.
    ============== =====================================
    """
    _tlv_parsers = {}

    def __init__(self, tlvs):
        super(lldp, self).__init__()
        self.tlvs = tlvs

    def _tlvs_len_valid(self):
        return len(self.tlvs) >= 4

    def _tlvs_valid(self):
        return self.tlvs[0].tlv_type == LLDP_TLV_CHASSIS_ID and self.tlvs[1].tlv_type == LLDP_TLV_PORT_ID and (self.tlvs[2].tlv_type == LLDP_TLV_TTL) and (self.tlvs[-1].tlv_type == LLDP_TLV_END)

    @classmethod
    def _parser(cls, buf):
        tlvs = []
        while buf:
            tlv_type = LLDPBasicTLV.get_type(buf)
            tlv = cls._tlv_parsers[tlv_type](buf)
            tlvs.append(tlv)
            offset = LLDP_TLV_SIZE + tlv.len
            buf = buf[offset:]
            if tlv.tlv_type == LLDP_TLV_END:
                break
            assert len(buf) > 0
        lldp_pkt = cls(tlvs)
        assert lldp_pkt._tlvs_len_valid()
        assert lldp_pkt._tlvs_valid()
        return (lldp_pkt, None, buf)

    @classmethod
    def parser(cls, buf):
        try:
            return cls._parser(buf)
        except:
            return (None, None, buf)

    def serialize(self, payload, prev):
        data = bytearray()
        for tlv in self.tlvs:
            data += tlv.serialize()
        return data

    @classmethod
    def set_type(cls, tlv_cls):
        cls._tlv_parsers[tlv_cls.tlv_type] = tlv_cls

    @classmethod
    def get_type(cls, tlv_type):
        return cls._tlv_parsers[tlv_type]

    @classmethod
    def set_tlv_type(cls, tlv_type):

        def _set_type(tlv_cls):
            tlv_cls.set_tlv_type(tlv_cls, tlv_type)
            cls.set_type(tlv_cls)
            return tlv_cls
        return _set_type

    def __len__(self):
        return sum((LLDP_TLV_SIZE + tlv.len for tlv in self.tlvs))