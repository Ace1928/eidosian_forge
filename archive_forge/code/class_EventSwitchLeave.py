import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventSwitchLeave(EventSwitchBase):

    def __init__(self, switch):
        super(EventSwitchLeave, self).__init__(switch)