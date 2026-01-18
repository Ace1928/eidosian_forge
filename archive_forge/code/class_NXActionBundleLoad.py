import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionBundleLoad(_NXActionBundleBase):
    """
        Select bundle link action

        This action has the same behavior as the bundle action,
        with one exception.
        Please refer to the ovs-ofctl command manual for details.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          bundle_load(fields, basis, algorithm, slave_type,
                      dst[start..end], slaves:[ s1, s2,...])
        ..

        +-----------------------------------------------------------+
        | **bundle_load(**\\ *fields*\\, \\ *basis*\\, \\ *algorithm*\\,  |
        | *slave_type*\\, \\ *dst*\\[\\ *start*\\... \\*emd*\\],           |
        | \\ *slaves*\\:[ \\ *s1*\\, \\ *s2*\\,...]\\ **)** |              |
        +-----------------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        algorithm        One of NX_MP_ALG_*.
        fields           One of NX_HASH_FIELDS_*
        basis            Universal hash parameter
        slave_type       Type of slaves(must be NXM_OF_IN_PORT)
        n_slaves         Number of slaves
        ofs_nbits        Start and End for the OXM/NXM field.
                         Setting method refer to the ``nicira_ext.ofs_nbits``
        dst              OXM/NXM header for source field
        slaves           List of slaves
        ================ ======================================================


        Example::

            actions += [parser.NXActionBundleLoad(
                            algorithm=nicira_ext.NX_MP_ALG_HRW,
                            fields=nicira_ext.NX_HASH_FIELDS_ETH_SRC,
                            basis=0,
                            slave_type=nicira_ext.NXM_OF_IN_PORT,
                            n_slaves=2,
                            ofs_nbits=nicira_ext.ofs_nbits(4, 31),
                            dst="reg0",
                            slaves=[2, 3])]
        """
    _subtype = nicira_ext.NXAST_BUNDLE_LOAD
    _TYPE = {'ascii': ['dst']}

    def __init__(self, algorithm, fields, basis, slave_type, n_slaves, ofs_nbits, dst, slaves):
        super(NXActionBundleLoad, self).__init__(algorithm, fields, basis, slave_type, n_slaves, ofs_nbits, dst, slaves)