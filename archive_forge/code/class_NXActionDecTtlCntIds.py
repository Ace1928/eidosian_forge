import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionDecTtlCntIds(NXAction):
    """
        Decrement TTL action

        This action decrements TTL of IPv4 packet or
        hop limits of IPv6 packet.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          dec_ttl(id1[,id2]...)
        ..

        +-------------------------------------------+
        | **dec_ttl(**\\ *id1*\\[,\\ *id2*\\]...\\ **)** |
        +-------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        cnt_ids          Controller ids
        ================ ======================================================

        Example::

            actions += [parser.NXActionDecTtlCntIds(cnt_ids=[1,2,3])]

        .. NOTE::
            If you want to set the following ovs-ofctl command.
            Please use ``OFPActionDecNwTtl``.

        +-------------+
        | **dec_ttl** |
        +-------------+
        """
    _subtype = nicira_ext.NXAST_DEC_TTL_CNT_IDS
    _fmt_str = '!H4x'
    _fmt_len = 6

    def __init__(self, cnt_ids, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionDecTtlCntIds, self).__init__()
        self.cnt_ids = cnt_ids

    @classmethod
    def parser(cls, buf):
        controllers, = struct.unpack_from(cls._fmt_str, buf)
        offset = cls._fmt_len
        cnt_ids = []
        for i in range(0, controllers):
            id_ = struct.unpack_from('!H', buf, offset)
            cnt_ids.append(id_[0])
            offset += 2
        return cls(cnt_ids)

    def serialize_body(self):
        assert isinstance(self.cnt_ids, (tuple, list))
        for i in self.cnt_ids:
            assert isinstance(i, int)
        controllers = len(self.cnt_ids)
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, controllers)
        offset = self._fmt_len
        for id_ in self.cnt_ids:
            msg_pack_into('!H', data, offset, id_)
            offset += 2
        id_len = utils.round_up(controllers, 4) - controllers
        if id_len != 0:
            msg_pack_into('%dx' % id_len * 2, data, offset)
        return data