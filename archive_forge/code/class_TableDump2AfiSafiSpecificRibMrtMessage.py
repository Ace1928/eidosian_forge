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
class TableDump2AfiSafiSpecificRibMrtMessage(TableDump2MrtMessage, metaclass=abc.ABCMeta):
    """
    MRT Message for the TABLE_DUMP_V2 Type and the AFI/SAFI-specific
    RIB subtypes.

    The AFI/SAFI-specific RIB subtypes consist of the RIB_IPV4_UNICAST,
    RIB_IPV4_MULTICAST, RIB_IPV6_UNICAST, RIB_IPV6_MULTICAST and their
    additional-path version subtypes.
    """
    _HEADER_FMT = '!I'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _PREFIX_CLS = None
    _IS_ADDPATH = False

    def __init__(self, seq_num, prefix, rib_entries, entry_count=None):
        self.seq_num = seq_num
        assert isinstance(prefix, self._PREFIX_CLS)
        self.prefix = prefix
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
        seq_num, = struct.unpack_from(cls._HEADER_FMT, buf)
        rest = buf[cls.HEADER_SIZE:]
        prefix, rest = cls._PREFIX_CLS.parser(rest)
        entry_count, rib_entries, _ = cls.parse_rib_entries(rest)
        return cls(seq_num, prefix, rib_entries, entry_count)

    def serialize_rib_entries(self):
        self.entry_count = len(self.rib_entries)
        rib_entries_bin = bytearray()
        for r in self.rib_entries:
            rib_entries_bin += r.serialize()
        return struct.pack('!H', self.entry_count) + rib_entries_bin

    def serialize(self):
        prefix_bin = self.prefix.serialize()
        rib_bin = self.serialize_rib_entries()
        return struct.pack(self._HEADER_FMT, self.seq_num) + prefix_bin + rib_bin