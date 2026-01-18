import abc
import logging
from os_ken.lib.packet import packet
def _packet_in_filter(packet_in_handler):

    def __packet_in_filter(self, ev):
        pkt = packet.Packet(ev.msg.data)
        if not packet_in_handler.pkt_in_filter.filter(pkt):
            if logging:
                LOG.debug('The packet is discarded by %s: %s', cls, pkt)
            return
        return packet_in_handler(self, ev)
    pkt_in_filter = cls(args)
    packet_in_handler.pkt_in_filter = pkt_in_filter
    return __packet_in_filter