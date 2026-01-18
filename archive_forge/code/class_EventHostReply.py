import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventHostReply(event.EventReplyBase):

    def __init__(self, dst, dpid, hosts):
        super(EventHostReply, self).__init__(dst)
        self.dpid = dpid
        self.hosts = hosts

    def __str__(self):
        return 'EventHostReply<dst=%s, dpid=%s, hosts=%s>' % (self.dst, self.dpid, len(self.hosts))