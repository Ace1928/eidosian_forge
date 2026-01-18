import abc
import logging
from os_ken.lib.packet import packet
class RequiredTypeFilter(PacketInFilterBase):

    def filter(self, pkt):
        required_types = self.args.get('types') or []
        for required_type in required_types:
            if not pkt.get_protocol(required_type):
                return False
        return True