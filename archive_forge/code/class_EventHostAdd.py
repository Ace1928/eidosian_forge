import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventHostAdd(EventHostBase):

    def __init__(self, host):
        super(EventHostAdd, self).__init__(host)