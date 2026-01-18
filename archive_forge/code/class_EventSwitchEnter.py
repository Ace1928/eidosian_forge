import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventSwitchEnter(EventSwitchBase):

    def __init__(self, switch):
        super(EventSwitchEnter, self).__init__(switch)