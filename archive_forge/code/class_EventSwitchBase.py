import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventSwitchBase(event.EventBase):

    def __init__(self, switch):
        super(EventSwitchBase, self).__init__()
        self.switch = switch

    def __str__(self):
        return '%s<dpid=%s, %s ports>' % (self.__class__.__name__, self.switch.dp.id, len(self.switch.ports))