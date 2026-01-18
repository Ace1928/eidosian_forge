import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@cfm.register_cfm_opcode(CFM_LOOPBACK_MESSAGE)
class loopback_message(loopback):
    """CFM (IEEE Std 802.1ag-2007) Loopback Message (LBM) encoder/decoder class.

    This is used with os_ken.lib.packet.cfm.cfm.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ================= =======================================
    Attribute         Description
    ================= =======================================
    md_lv             Maintenance Domain Level.
    version           The protocol version number.
    transaction_id    Loopback Transaction Identifier.
    tlvs              TLVs.
    ================= =======================================
    """

    def __init__(self, md_lv=0, version=CFM_VERSION, transaction_id=0, tlvs=None):
        super(loopback_message, self).__init__(md_lv, version, transaction_id, tlvs)
        self._opcode = CFM_LOOPBACK_MESSAGE