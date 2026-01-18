from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
class EventRowBase(os_ken_event.EventBase):

    def __init__(self, system_id, table, row, event_type):
        super(EventRowBase, self).__init__()
        self.system_id = system_id
        self.table = table
        self.row = row
        self.event_type = event_type

    def __str__(self):
        return '%s<system_id=%s table=%s, uuid=%s>' % (self.__class__.__name__, self.system_id, self.table, self.row['_uuid'])