import copy
import io
import json
import testtools
from urllib import parse
from glanceclient.v2 import schemas
class FakeNoTTYStdout(FakeTTYStdout):
    """A Fake stdout that is not a TTY device."""

    def isatty(self):
        return False