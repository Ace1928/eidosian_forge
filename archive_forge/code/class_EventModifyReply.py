from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
class EventModifyReply(os_ken_event.EventReplyBase):

    def __init__(self, system_id, status, insert_uuids, err_msg):
        self.system_id = system_id
        self.status = status
        self.insert_uuids = insert_uuids
        self.err_msg = err_msg

    def __str__(self):
        return '%s<system_id=%s, status=%s, insert_uuids=%s, error_msg=%s>' % (self.__class__.__name__, self.system_id, self.status, self.insert_uuids, self.err_msg)