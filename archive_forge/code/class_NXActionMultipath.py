import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionMultipath(NXAction):
    """
        Select multipath link action

        This action selects multipath link based on the specified parameters.
        Please refer to the ovs-ofctl command manual for details.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          multipath(fields, basis, algorithm, n_links, arg, dst[start..end])
        ..

        +-------------------------------------------------------------+
        | **multipath(**\\ *fields*\\, \\ *basis*\\, \\ *algorithm*\\,      |
        | *n_links*\\, \\ *arg*\\, \\ *dst*\\[\\ *start*\\..\\ *end*\\]\\ **)** |
        +-------------------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        fields           One of NX_HASH_FIELDS_*
        basis            Universal hash parameter
        algorithm        One of NX_MP_ALG_*.
        max_link         Number of output links
        arg              Algorithm-specific argument
        ofs_nbits        Start and End for the OXM/NXM field.
                         Setting method refer to the ``nicira_ext.ofs_nbits``
        dst              OXM/NXM header for source field
        ================ ======================================================

        Example::

            actions += [parser.NXActionMultipath(
                            fields=nicira_ext.NX_HASH_FIELDS_SYMMETRIC_L4,
                            basis=1024,
                            algorithm=nicira_ext.NX_MP_ALG_HRW,
                            max_link=5,
                            arg=0,
                            ofs_nbits=nicira_ext.ofs_nbits(4, 31),
                            dst="reg2")]
        """
    _subtype = nicira_ext.NXAST_MULTIPATH
    _fmt_str = '!HH2xHHI2xH4s'
    _TYPE = {'ascii': ['dst']}

    def __init__(self, fields, basis, algorithm, max_link, arg, ofs_nbits, dst, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionMultipath, self).__init__()
        self.fields = fields
        self.basis = basis
        self.algorithm = algorithm
        self.max_link = max_link
        self.arg = arg
        self.ofs_nbits = ofs_nbits
        self.dst = dst

    @classmethod
    def parser(cls, buf):
        fields, basis, algorithm, max_link, arg, ofs_nbits, oxm_data = struct.unpack_from(cls._fmt_str, buf, 0)
        n, len_ = ofp.oxm_parse_header(oxm_data, 0)
        dst = ofp.oxm_to_user_header(n)
        return cls(fields, basis, algorithm, max_link, arg, ofs_nbits, dst)

    def serialize_body(self):
        data = bytearray()
        dst = bytearray()
        oxm = ofp.oxm_from_user_header(self.dst)
        (ofp.oxm_serialize_header(oxm, dst, 0),)
        msg_pack_into(self._fmt_str, data, 0, self.fields, self.basis, self.algorithm, self.max_link, self.arg, self.ofs_nbits, bytes(dst))
        return data