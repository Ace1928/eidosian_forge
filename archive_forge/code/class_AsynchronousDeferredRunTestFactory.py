import io
import warnings
import sys
from fixtures import CompoundFixture, Fixture
from testtools.content import Content, text_content
from testtools.content_type import UTF8_TEXT
from testtools.runtest import RunTest, _raise_force_fail_error
from ._deferred import extract_result
from ._spinner import (
from twisted.internet import defer
from twisted.python import log
class AsynchronousDeferredRunTestFactory:

    def __call__(self, case, handlers=None, last_resort=None):
        return cls(case, handlers, last_resort, reactor, timeout, debug, suppress_twisted_logging, store_twisted_logs)