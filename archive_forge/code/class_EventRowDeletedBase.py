from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
class EventRowDeletedBase(EventRowDelete):

    def __init__(self, ev):
        super(EventRowDeletedBase, self).__init__(ev.system_id, ev.table, ev.row)