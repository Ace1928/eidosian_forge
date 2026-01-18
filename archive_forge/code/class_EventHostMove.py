import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventHostMove(event.EventBase):

    def __init__(self, src, dst):
        super(EventHostMove, self).__init__()
        self.src = src
        self.dst = dst

    def __str__(self):
        return '%s<src=%s, dst=%s>' % (self.__class__.__name__, self.src, self.dst)