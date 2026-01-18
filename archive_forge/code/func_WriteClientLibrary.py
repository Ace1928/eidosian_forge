import datetime
from apitools.gen import message_registry
from apitools.gen import service_registry
from apitools.gen import util
def WriteClientLibrary(self, out):
    self.__services_registry.WriteFile(self._GetPrinter(out))