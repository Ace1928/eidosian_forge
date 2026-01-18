import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
class TestStreamResultToBytesContract(TestCase, TestStreamResultContract):
    """Check that StreamResult behaves as testtools expects."""

    def _make_result(self):
        return subunit.StreamResultToBytes(BytesIO())