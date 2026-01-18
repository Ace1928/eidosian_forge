import datetime
import optparse
from contextlib import contextmanager
from functools import partial
from io import BytesIO, TextIOWrapper
from tempfile import NamedTemporaryFile
from iso8601 import UTC
from testtools import TestCase
from testtools.matchers import (Equals, Matcher, MatchesAny, MatchesListwise,
from testtools.testresult.doubles import StreamResult
import subunit._output as _o
from subunit._output import (_ALL_ACTIONS, _FINAL_ACTIONS,
class SafeOptionParser(optparse.OptionParser):
    """An ArgumentParser class that doesn't call sys.exit."""

    def exit(self, status=0, message=''):
        raise RuntimeError(message)

    def error(self, message):
        raise RuntimeError(message)