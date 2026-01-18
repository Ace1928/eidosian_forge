import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventLinkReply(event.EventReplyBase):

    def __init__(self, dst, dpid, links):
        super(EventLinkReply, self).__init__(dst)
        self.dpid = dpid
        self.links = links

    def __str__(self):
        return 'EventLinkReply<dst=%s, dpid=%s, links=%s>' % (self.dst, self.dpid, len(self.links))