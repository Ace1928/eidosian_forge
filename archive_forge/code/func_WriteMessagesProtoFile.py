import datetime
from apitools.gen import message_registry
from apitools.gen import service_registry
from apitools.gen import util
def WriteMessagesProtoFile(self, out):
    self.__message_registry.WriteProtoFile(self._GetPrinter(out))