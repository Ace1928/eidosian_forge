import random
import email.message
import pyzor
class ClientSideRequest(Request):
    op = None

    def setup(self):
        Request.setup(self)
        self['Op'] = self.op