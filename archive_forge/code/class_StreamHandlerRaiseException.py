from logging import StreamHandler, getLogger, INFO, Formatter
import sys
from fixtures import Fixture
from fixtures._fixtures.streams import StringStream
class StreamHandlerRaiseException(StreamHandler):
    """Handler class that will raise an exception on formatting errors."""

    def handleError(self, record):
        _, value, tb = sys.exc_info()
        raise value.with_traceback(tb)