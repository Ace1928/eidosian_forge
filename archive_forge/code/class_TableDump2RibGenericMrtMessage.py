import abc
import logging
import struct
import time
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from os_ken.lib.packet import bgp
from os_ken.lib.packet import ospf
@TableDump2MrtMessage.register_type(TableDump2MrtRecord.SUBTYPE_RIB_GENERIC)
class TableDump2RibGenericMrtMessage(TableDump2MrtMessage):
    """
    MRT Message for the TABLE_DUMP_V2 Type and the generic RIB subtypes.

    The generic RIB subtypes consist of the RIB_GENERIC and
    RIB_GENERIC_ADDPATH subtypes.
    """
    _HEADER_FMT = '!IHB'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _IS_ADDPATH = False

    def __init__(self, seq_num, afi, safi, nlri, rib_entries, entry_count=None):
        self.seq_num = seq_num
        self.afi = afi
        self.safi = safi
        assert isinstance(nlri, bgp._AddrPrefix)
        self.nlri = nlri
        self.entry_count = entry_count
        assert isinstance(rib_entries, (list, tuple))
        for rib_entry in rib_entries:
            assert isinstance(rib_entry, MrtRibEntry)
        self.rib_entries = rib_entries

    @classmethod
    def parse_rib_entries(cls, buf):
        entry_count, = struct.unpack_from('!H', buf)
        rest = buf[2:]
        rib_entries = []
        for i in range(entry_count):
            r, rest = MrtRibEntry.parse(rest, is_addpath=cls._IS_ADDPATH)
            rib_entries.insert(i, r)
        return (entry_count, rib_entries, rest)

    @classmethod
    def parse(cls, buf):
        seq_num, afi, safi = struct.unpack_from(cls._HEADER_FMT, buf)
        rest = buf[cls.HEADER_SIZE:]
        nlri, rest = bgp.BGPNLRI.parser(rest)
        entry_count, rib_entries, _ = cls.parse_rib_entries(rest)
        return cls(seq_num, afi, safi, nlri, rib_entries, entry_count)

    def serialize_rib_entries(self):
        self.entry_count = len(self.rib_entries)
        rib_entries_bin = bytearray()
        for r in self.rib_entries:
            rib_entries_bin += r.serialize()
        return struct.pack('!H', self.entry_count) + rib_entries_bin

    def serialize(self):
        nlri_bin = self.nlri.serialize()
        rib_bin = self.serialize_rib_entries()
        return struct.pack(self._HEADER_FMT, self.seq_num, self.afi, self.safi) + nlri_bin + rib_bin