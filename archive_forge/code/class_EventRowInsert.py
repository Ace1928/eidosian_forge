from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
class EventRowInsert(EventRowBase):

    def __init__(self, system_id, table, row):
        super(EventRowInsert, self).__init__(system_id, table, row, 'Inserted')