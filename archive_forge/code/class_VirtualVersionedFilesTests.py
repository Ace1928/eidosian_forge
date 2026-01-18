import itertools
from gzip import GzipFile
from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils, progress, transport, ui
from ...errors import RevisionAlreadyPresent, RevisionNotPresent
from ...tests import (TestCase, TestCaseWithMemoryTransport, TestNotApplicable,
from ...tests.http_utils import TestCaseWithWebserver
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from .. import groupcompress
from .. import knit as _mod_knit
from .. import versionedfile as versionedfile
from ..knit import cleanup_pack_knit, make_file_factory, make_pack_factory
from ..versionedfile import (ChunkedContentFactory, ConstantMapper,
from ..weave import WeaveFile, WeaveInvalidChecksum
from ..weavefile import write_weave
class VirtualVersionedFilesTests(TestCase):
    """Basic tests for the VirtualVersionedFiles implementations."""

    def _get_parent_map(self, keys):
        ret = {}
        for k in keys:
            if k in self._parent_map:
                ret[k] = self._parent_map[k]
        return ret

    def setUp(self):
        super().setUp()
        self._lines = {}
        self._parent_map = {}
        self.texts = VirtualVersionedFiles(self._get_parent_map, self._lines.get)

    def test_add_lines(self):
        self.assertRaises(NotImplementedError, self.texts.add_lines, b'foo', [], [])

    def test_add_mpdiffs(self):
        self.assertRaises(NotImplementedError, self.texts.add_mpdiffs, [])

    def test_check_noerrors(self):
        self.texts.check()

    def test_insert_record_stream(self):
        self.assertRaises(NotImplementedError, self.texts.insert_record_stream, [])

    def test_get_sha1s_nonexistent(self):
        self.assertEqual({}, self.texts.get_sha1s([(b'NONEXISTENT',)]))

    def test_get_sha1s(self):
        self._lines[b'key'] = [b'dataline1', b'dataline2']
        self.assertEqual({(b'key',): osutils.sha_strings(self._lines[b'key'])}, self.texts.get_sha1s([(b'key',)]))

    def test_get_parent_map(self):
        self._parent_map = {b'G': (b'A', b'B')}
        self.assertEqual({(b'G',): ((b'A',), (b'B',))}, self.texts.get_parent_map([(b'G',), (b'L',)]))

    def test_get_record_stream(self):
        self._lines[b'A'] = [b'FOO', b'BAR']
        it = self.texts.get_record_stream([(b'A',)], 'unordered', True)
        record = next(it)
        self.assertEqual('chunked', record.storage_kind)
        self.assertEqual(b'FOOBAR', record.get_bytes_as('fulltext'))
        self.assertEqual([b'FOO', b'BAR'], record.get_bytes_as('chunked'))

    def test_get_record_stream_absent(self):
        it = self.texts.get_record_stream([(b'A',)], 'unordered', True)
        record = next(it)
        self.assertEqual('absent', record.storage_kind)

    def test_iter_lines_added_or_present_in_keys(self):
        self._lines[b'A'] = [b'FOO', b'BAR']
        self._lines[b'B'] = [b'HEY']
        self._lines[b'C'] = [b'Alberta']
        it = self.texts.iter_lines_added_or_present_in_keys([(b'A',), (b'B',)])
        self.assertEqual(sorted([(b'FOO', b'A'), (b'BAR', b'A'), (b'HEY', b'B')]), sorted(list(it)))