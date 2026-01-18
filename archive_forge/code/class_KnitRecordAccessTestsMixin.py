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
class KnitRecordAccessTestsMixin:
    """Tests for getting and putting knit records."""

    def test_add_raw_records(self):
        """add_raw_records adds records retrievable later."""
        access = self.get_access()
        memos = access.add_raw_records([(b'key', 10)], [b'1234567890'])
        self.assertEqual([b'1234567890'], list(access.get_raw_records(memos)))

    def test_add_raw_record(self):
        """add_raw_record adds records retrievable later."""
        access = self.get_access()
        memos = access.add_raw_record(b'key', 10, [b'1234567890'])
        self.assertEqual([b'1234567890'], list(access.get_raw_records([memos])))

    def test_add_several_raw_records(self):
        """add_raw_records with many records and read some back."""
        access = self.get_access()
        memos = access.add_raw_records([(b'key', 10), (b'key2', 2), (b'key3', 5)], [b'12345678901234567'])
        self.assertEqual([b'1234567890', b'12', b'34567'], list(access.get_raw_records(memos)))
        self.assertEqual([b'1234567890'], list(access.get_raw_records(memos[0:1])))
        self.assertEqual([b'12'], list(access.get_raw_records(memos[1:2])))
        self.assertEqual([b'34567'], list(access.get_raw_records(memos[2:3])))
        self.assertEqual([b'1234567890', b'34567'], list(access.get_raw_records(memos[0:1] + memos[2:3])))