from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
class EventReadRequest(os_ken_event.EventRequestBase):

    def __init__(self, system_id, func):
        self.system_id = system_id
        self.func = func
        self.dst = 'OVSDB'