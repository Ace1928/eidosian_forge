import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
class TestKnitKnitAccess(TestCaseWithMemoryTransport, KnitRecordAccessTestsMixin):
    """Tests for the .kndx implementation."""

    def get_access(self):
        """Get a .knit style access instance."""
        mapper = ConstantMapper('foo')
        access = _KnitKeyAccess(self.get_transport(), mapper)
        return access