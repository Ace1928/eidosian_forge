import random
import email.message
import pyzor
class ThreadedMessage(Message):

    def init_for_sending(self):
        if 'Thread' not in self:
            self.set_thread(ThreadId.generate())
        assert 'Thread' in self
        self['PV'] = str(pyzor.proto_version)
        Message.init_for_sending(self)

    def ensure_complete(self):
        if 'PV' not in self or 'Thread' not in self:
            raise pyzor.IncompleteMessageError("Doesn't have fields for a ThreadedMessage.")
        Message.ensure_complete(self)

    def get_protocol_version(self):
        return float(self['PV'])

    def get_thread(self):
        return ThreadId(self['Thread'])

    def set_thread(self, i):
        self['Thread'] = str(i)