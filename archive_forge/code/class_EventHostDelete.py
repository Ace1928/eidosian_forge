import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventHostDelete(EventHostBase):

    def __init__(self, host):
        super(EventHostDelete, self).__init__(host)