import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionStackPush(NXActionStackBase):
    """
        Push field action

        This action pushes field to top of the stack.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          pop:dst[start...end]
        ..

        +----------------------------------------------------+
        | **pop**\\:\\ *dst*\\ **[**\\ *start*\\...\\ *end*\\ **]** |
        +----------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        field            OXM/NXM header for source field
        start            Start bit for source field
        end              End bit for source field
        ================ ======================================================

        Example::

            actions += [parser.NXActionStackPush(field="reg2",
                                                 start=0,
                                                 end=5)]
        """
    _subtype = nicira_ext.NXAST_STACK_PUSH