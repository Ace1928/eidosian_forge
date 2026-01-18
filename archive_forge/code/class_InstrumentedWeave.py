from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
class InstrumentedWeave(Weave):
    """Keep track of how many times functions are called."""

    def __init__(self, weave_name=None):
        self._extract_count = 0
        Weave.__init__(self, weave_name=weave_name)

    def _extract(self, versions):
        self._extract_count += 1
        return Weave._extract(self, versions)