import datetime
from apitools.gen import message_registry
from apitools.gen import service_registry
from apitools.gen import util
def _GetPrinter(self, out):
    printer = util.SimplePrettyPrinter(out)
    return printer