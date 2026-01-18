import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventSwitchReconnected(EventSwitchBase):

    def __init__(self, switch):
        super(EventSwitchReconnected, self).__init__(switch)