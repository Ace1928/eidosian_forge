import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionSetQueue(NXAction):
    """
        Set queue action

        This action sets the queue that should be used to queue
        when packets are output.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          set_queue:queue
        ..

        +-------------------------+
        | **set_queue**\\:\\ *queue*|
        +-------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        queue_id         Queue ID for the packets
        ================ ======================================================

        .. note::
            This actions is supported by
            ``OFPActionSetQueue``
            in OpenFlow1.2 or later.

        Example::

            actions += [parser.NXActionSetQueue(queue_id=10)]
        """
    _subtype = nicira_ext.NXAST_SET_QUEUE
    _fmt_str = '!2xI'

    def __init__(self, queue_id, type_=None, len_=None, vendor=None, subtype=None):
        super(NXActionSetQueue, self).__init__()
        self.queue_id = queue_id

    @classmethod
    def parser(cls, buf):
        queue_id, = struct.unpack_from(cls._fmt_str, buf, 0)
        return cls(queue_id)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.queue_id)
        return data