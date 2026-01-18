import datetime
from apitools.gen import message_registry
from apitools.gen import service_registry
from apitools.gen import util
def WriteMessagesFile(self, out):
    self.__message_registry.WriteFile(self._GetPrinter(out))