import numbers
from os_ken.controller import event
class SendMsgRequest(_RequestBase):

    def __init__(self, msg, reply_cls=None, reply_multi=False):
        super(SendMsgRequest, self).__init__()
        self.msg = msg
        self.reply_cls = reply_cls
        self.reply_multi = reply_multi