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
class TestKnitIndex(KnitTests):

    def test_add_versions_dictionary_compresses(self):
        """Adding versions to the index should update the lookup dict"""
        knit = self.make_test_knit()
        idx = knit._index
        idx.add_records([((b'a-1',), [b'fulltext'], ((b'a-1',), 0, 0), [])])
        self.check_file_contents('test.kndx', b'# bzr knit index 8\n\na-1 fulltext 0 0  :')
        idx.add_records([((b'a-2',), [b'fulltext'], ((b'a-2',), 0, 0), [(b'a-1',)]), ((b'a-3',), [b'fulltext'], ((b'a-3',), 0, 0), [(b'a-2',)])])
        self.check_file_contents('test.kndx', b'# bzr knit index 8\n\na-1 fulltext 0 0  :\na-2 fulltext 0 0 0 :\na-3 fulltext 0 0 1 :')
        self.assertEqual({(b'a-3',), (b'a-1',), (b'a-2',)}, idx.keys())
        self.assertEqual({(b'a-1',): (((b'a-1',), 0, 0), None, (), ('fulltext', False)), (b'a-2',): (((b'a-2',), 0, 0), None, ((b'a-1',),), ('fulltext', False)), (b'a-3',): (((b'a-3',), 0, 0), None, ((b'a-2',),), ('fulltext', False))}, idx.get_build_details(idx.keys()))
        self.assertEqual({(b'a-1',): (), (b'a-2',): ((b'a-1',),), (b'a-3',): ((b'a-2',),)}, idx.get_parent_map(idx.keys()))

    def test_add_versions_fails_clean(self):
        """If add_versions fails in the middle, it restores a pristine state.

        Any modifications that are made to the index are reset if all versions
        cannot be added.
        """
        knit = self.make_test_knit()
        idx = knit._index
        idx.add_records([((b'a-1',), [b'fulltext'], ((b'a-1',), 0, 0), [])])

        class StopEarly(Exception):
            pass

        def generate_failure():
            """Add some entries and then raise an exception"""
            yield ((b'a-2',), [b'fulltext'], (None, 0, 0), (b'a-1',))
            yield ((b'a-3',), [b'fulltext'], (None, 0, 0), (b'a-2',))
            raise StopEarly()

        def assertA1Only():
            self.assertEqual({(b'a-1',)}, set(idx.keys()))
            self.assertEqual({(b'a-1',): (((b'a-1',), 0, 0), None, (), ('fulltext', False))}, idx.get_build_details([(b'a-1',)]))
            self.assertEqual({(b'a-1',): ()}, idx.get_parent_map(idx.keys()))
        assertA1Only()
        self.assertRaises(StopEarly, idx.add_records, generate_failure())
        assertA1Only()

    def test_knit_index_ignores_empty_files(self):
        t = _mod_transport.get_transport_from_path('.')
        t.put_bytes('test.kndx', b'')
        knit = self.make_test_knit()

    def test_knit_index_checks_header(self):
        t = _mod_transport.get_transport_from_path('.')
        t.put_bytes('test.kndx', b'# not really a knit header\n\n')
        k = self.make_test_knit()
        self.assertRaises(KnitHeaderError, k.keys)