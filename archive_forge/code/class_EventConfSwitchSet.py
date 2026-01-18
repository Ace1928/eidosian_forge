import logging
from os_ken.controller import event
from os_ken.lib.dpid import dpid_to_str
from os_ken.base import app_manager
class EventConfSwitchSet(event.EventBase):

    def __init__(self, dpid, key, value):
        super(EventConfSwitchSet, self).__init__()
        self.dpid = dpid
        self.key = key
        self.value = value

    def __str__(self):
        return 'EventConfSwitchSet<%s, %s, %s>' % (dpid_to_str(self.dpid), self.key, self.value)