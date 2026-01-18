import gzip
import os
from io import BytesIO
from ... import errors as errors
from ... import transactions, transport
from ...bzr.weave import WeaveFile
from ...errors import BzrError
from ...tests import TestCase, TestCaseInTempDir, TestCaseWithTransport
from ...transport.memory import MemoryTransport
from .store import TransportStore
from .store.text import TextStore
from .store.versioned import VersionedFileStore
class MockTransport(transport.Transport):
    """A fake transport for testing with."""

    def has(self, filename):
        return False

    def __init__(self, url=None):
        if url is None:
            url = 'http://example.com'
        super().__init__(url)

    def mkdir(self, filename):
        return