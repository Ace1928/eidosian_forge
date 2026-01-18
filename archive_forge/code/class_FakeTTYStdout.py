import copy
import io
import json
import testtools
from urllib import parse
from glanceclient.v2 import schemas
class FakeTTYStdout(io.StringIO):
    """A Fake stdout that try to emulate a TTY device as much as possible."""

    def isatty(self):
        return True

    def write(self, data):
        if data.startswith('\r'):
            self.seek(0)
            data = data[1:]
        return io.StringIO.write(self, data)