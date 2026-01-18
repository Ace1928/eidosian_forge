from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.core import log
def SetToDefaultInstance(self, message_class):
    self.message = message_class()
    self.message_class = message_class