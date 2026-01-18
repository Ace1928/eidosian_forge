from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
class EventReadReply(os_ken_event.EventReplyBase):

    def __init__(self, system_id, result, err_msg=''):
        self.system_id = system_id
        self.result = result
        self.err_msg = err_msg