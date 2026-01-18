import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class cfm(packet_base.PacketBase):
    """CFM (Connectivity Fault Management) Protocol header class.

    http://standards.ieee.org/getieee802/download/802.1ag-2007.pdf

    OpCode Field range assignments

    +---------------+--------------------------------------------------+
    | OpCode range  | CFM PDU or organization                          |
    +===============+==================================================+
    | 0             | Reserved for IEEE 802.1                          |
    +---------------+--------------------------------------------------+
    | 1             | Continuity Check Message (CCM)                   |
    +---------------+--------------------------------------------------+
    | 2             | Loopback Reply (LBR)                             |
    +---------------+--------------------------------------------------+
    | 3             | Loopback Message (LBM)                           |
    +---------------+--------------------------------------------------+
    | 4             | Linktrace Reply (LTR)                            |
    +---------------+--------------------------------------------------+
    | 5             | Linktrace Message (LTM)                          |
    +---------------+--------------------------------------------------+
    | 06 - 31       | Reserved for IEEE 802.1                          |
    +---------------+--------------------------------------------------+
    | 32 - 63       | Defined by ITU-T Y.1731                          |
    +---------------+--------------------------------------------------+
    | 64 - 255      | Reserved for IEEE 802.1.                         |
    +---------------+--------------------------------------------------+

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== ========================================
    Attribute      Description
    ============== ========================================
    op             CFM PDU
    ============== ========================================

    """
    _PACK_STR = '!B'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _CFM_OPCODE = {}
    _TYPE = {'ascii': ['ltm_orig_addr', 'ltm_targ_addr']}

    @staticmethod
    def register_cfm_opcode(type_):

        def _register_cfm_opcode(cls):
            cfm._CFM_OPCODE[type_] = cls
            return cls
        return _register_cfm_opcode

    def __init__(self, op=None):
        super(cfm, self).__init__()
        assert isinstance(op, operation)
        self.op = op

    @classmethod
    def parser(cls, buf):
        opcode, = struct.unpack_from(cls._PACK_STR, buf, cls._MIN_LEN)
        cls_ = cls._CFM_OPCODE.get(opcode)
        op = cls_.parser(buf)
        instance = cls(op)
        rest = buf[len(instance):]
        return (instance, None, rest)

    def serialize(self, payload, prev):
        buf = self.op.serialize()
        return buf

    def __len__(self):
        return len(self.op)