import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class link_trace(operation):

    @abc.abstractmethod
    def __init__(self, md_lv, version, use_fdb_only, transaction_id, ttl, tlvs):
        super(link_trace, self).__init__(md_lv, version, tlvs)
        assert use_fdb_only in [0, 1]
        self.use_fdb_only = use_fdb_only
        self.transaction_id = transaction_id
        self.ttl = ttl

    @classmethod
    @abc.abstractmethod
    def parser(cls, buf):
        pass

    @abc.abstractmethod
    def serialize(self):
        pass

    def __len__(self):
        return self._calc_len(self._MIN_LEN)