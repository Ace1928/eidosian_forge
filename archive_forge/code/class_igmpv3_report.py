import struct
from math import trunc
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
class igmpv3_report(igmp):
    """
    Internet Group Management Protocol(IGMP, RFC 3376)
    Membership Report message encoder/decoder class.

    http://www.ietf.org/rfc/rfc3376.txt

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    =============== ====================================================
    Attribute       Description
    =============== ====================================================
    msgtype         a message type for v3.
    csum            a check sum value. 0 means automatically-calculate
                    when encoding.
    record_num      a number of the group records.
    records         a list of os_ken.lib.packet.igmp.igmpv3_report_group.
                    None if no records.
    =============== ====================================================
    """
    _PACK_STR = '!BxH2xH'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _class_prefixes = ['igmpv3_report_group']

    def __init__(self, msgtype=IGMP_TYPE_REPORT_V3, csum=0, record_num=0, records=None):
        self.msgtype = msgtype
        self.csum = csum
        self.record_num = record_num
        records = records or []
        assert isinstance(records, list)
        for record in records:
            assert isinstance(record, igmpv3_report_group)
        self.records = records

    @classmethod
    def parser(cls, buf):
        msgtype, csum, record_num = struct.unpack_from(cls._PACK_STR, buf)
        offset = cls._MIN_LEN
        records = []
        while 0 < len(buf[offset:]) and record_num > len(records):
            record = igmpv3_report_group.parser(buf[offset:])
            records.append(record)
            offset += len(record)
        assert record_num == len(records)
        return (cls(msgtype, csum, record_num, records), None, buf[offset:])

    def serialize(self, payload, prev):
        buf = bytearray(struct.pack(self._PACK_STR, self.msgtype, self.csum, self.record_num))
        for record in self.records:
            buf.extend(record.serialize())
        if 0 == self.record_num:
            self.record_num = len(self.records)
            struct.pack_into('!H', buf, 6, self.record_num)
        if 0 == self.csum:
            self.csum = packet_utils.checksum(buf)
            struct.pack_into('!H', buf, 2, self.csum)
        return bytes(buf)

    def __len__(self):
        records_len = 0
        for record in self.records:
            records_len += len(record)
        return self._MIN_LEN + records_len