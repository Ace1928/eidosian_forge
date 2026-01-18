import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventHostBase(event.EventBase):

    def __init__(self, host):
        super(EventHostBase, self).__init__()
        self.host = host

    def __str__(self):
        return '%s<%s>' % (self.__class__.__name__, self.host)