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
class TestVersionedFiles(TestCaseWithMemoryTransport):
    """Tests for the multiple-file variant of VersionedFile."""
    len_one_scenarios = [('weave-named', {'cleanup': None, 'factory': make_versioned_files_factory(WeaveFile, ConstantMapper('inventory')), 'graph': True, 'key_length': 1, 'support_partial_insertion': False}), ('named-knit', {'cleanup': None, 'factory': make_file_factory(False, ConstantMapper('revisions')), 'graph': True, 'key_length': 1, 'support_partial_insertion': False}), ('named-nograph-nodelta-knit-pack', {'cleanup': cleanup_pack_knit, 'factory': make_pack_factory(False, False, 1), 'graph': False, 'key_length': 1, 'support_partial_insertion': False}), ('named-graph-knit-pack', {'cleanup': cleanup_pack_knit, 'factory': make_pack_factory(True, True, 1), 'graph': True, 'key_length': 1, 'support_partial_insertion': True}), ('named-graph-nodelta-knit-pack', {'cleanup': cleanup_pack_knit, 'factory': make_pack_factory(True, False, 1), 'graph': True, 'key_length': 1, 'support_partial_insertion': False}), ('groupcompress-nograph', {'cleanup': groupcompress.cleanup_pack_group, 'factory': groupcompress.make_pack_factory(False, False, 1), 'graph': False, 'key_length': 1, 'support_partial_insertion': False})]
    len_two_scenarios = [('weave-prefix', {'cleanup': None, 'factory': make_versioned_files_factory(WeaveFile, PrefixMapper()), 'graph': True, 'key_length': 2, 'support_partial_insertion': False}), ('annotated-knit-escape', {'cleanup': None, 'factory': make_file_factory(True, HashEscapedPrefixMapper()), 'graph': True, 'key_length': 2, 'support_partial_insertion': False}), ('plain-knit-pack', {'cleanup': cleanup_pack_knit, 'factory': make_pack_factory(True, True, 2), 'graph': True, 'key_length': 2, 'support_partial_insertion': True}), ('groupcompress', {'cleanup': groupcompress.cleanup_pack_group, 'factory': groupcompress.make_pack_factory(True, False, 1), 'graph': True, 'key_length': 1, 'support_partial_insertion': False})]
    scenarios = len_one_scenarios + len_two_scenarios

    def get_versionedfiles(self, relpath='files'):
        transport = self.get_transport(relpath)
        if relpath != '.':
            transport.mkdir('.')
        files = self.factory(transport)
        if self.cleanup is not None:
            self.addCleanup(self.cleanup, files)
        return files

    def get_simple_key(self, suffix):
        """Return a key for the object under test."""
        if self.key_length == 1:
            return (suffix,)
        else:
            return (b'FileA',) + (suffix,)

    def test_add_fallback_implies_without_fallbacks(self):
        f = self.get_versionedfiles('files')
        if getattr(f, 'add_fallback_versioned_files', None) is None:
            raise TestNotApplicable("%s doesn't support fallbacks" % (f.__class__.__name__,))
        g = self.get_versionedfiles('fallback')
        key_a = self.get_simple_key(b'a')
        g.add_lines(key_a, [], [b'\n'])
        f.add_fallback_versioned_files(g)
        self.assertTrue(key_a in f.get_parent_map([key_a]))
        self.assertFalse(key_a in f.without_fallbacks().get_parent_map([key_a]))

    def test_add_lines(self):
        f = self.get_versionedfiles()
        key0 = self.get_simple_key(b'r0')
        key1 = self.get_simple_key(b'r1')
        key2 = self.get_simple_key(b'r2')
        keyf = self.get_simple_key(b'foo')
        f.add_lines(key0, [], [b'a\n', b'b\n'])
        if self.graph:
            f.add_lines(key1, [key0], [b'b\n', b'c\n'])
        else:
            f.add_lines(key1, [], [b'b\n', b'c\n'])
        keys = f.keys()
        self.assertTrue(key0 in keys)
        self.assertTrue(key1 in keys)
        records = []
        for record in f.get_record_stream([key0, key1], 'unordered', True):
            records.append((record.key, record.get_bytes_as('fulltext')))
        records.sort()
        self.assertEqual([(key0, b'a\nb\n'), (key1, b'b\nc\n')], records)

    def test_add_chunks(self):
        f = self.get_versionedfiles()
        key0 = self.get_simple_key(b'r0')
        key1 = self.get_simple_key(b'r1')
        key2 = self.get_simple_key(b'r2')
        keyf = self.get_simple_key(b'foo')

        def add_chunks(key, parents, chunks):
            factory = ChunkedContentFactory(key, parents, osutils.sha_strings(chunks), chunks)
            return f.add_content(factory)
        add_chunks(key0, [], [b'a', b'\nb\n'])
        if self.graph:
            add_chunks(key1, [key0], [b'b', b'\n', b'c\n'])
        else:
            add_chunks(key1, [], [b'b\n', b'c\n'])
        keys = f.keys()
        self.assertIn(key0, keys)
        self.assertIn(key1, keys)
        records = []
        for record in f.get_record_stream([key0, key1], 'unordered', True):
            records.append((record.key, record.get_bytes_as('fulltext')))
        records.sort()
        self.assertEqual([(key0, b'a\nb\n'), (key1, b'b\nc\n')], records)

    def test_annotate(self):
        files = self.get_versionedfiles()
        self.get_diamond_files(files)
        if self.key_length == 1:
            prefix = ()
        else:
            prefix = (b'FileA',)
        origins = files.annotate(prefix + (b'origin',))
        self.assertEqual([(prefix + (b'origin',), b'origin\n')], origins)
        origins = files.annotate(prefix + (b'base',))
        self.assertEqual([(prefix + (b'base',), b'base\n')], origins)
        origins = files.annotate(prefix + (b'merged',))
        if self.graph:
            self.assertEqual([(prefix + (b'base',), b'base\n'), (prefix + (b'left',), b'left\n'), (prefix + (b'right',), b'right\n'), (prefix + (b'merged',), b'merged\n')], origins)
        else:
            self.assertEqual([(prefix + (b'merged',), b'base\n'), (prefix + (b'merged',), b'left\n'), (prefix + (b'merged',), b'right\n'), (prefix + (b'merged',), b'merged\n')], origins)
        self.assertRaises(RevisionNotPresent, files.annotate, prefix + ('missing-key',))

    def test_check_no_parameters(self):
        files = self.get_versionedfiles()

    def test_check_progressbar_parameter(self):
        """A progress bar can be supplied because check can be a generator."""
        pb = ui.ui_factory.nested_progress_bar()
        self.addCleanup(pb.finished)
        files = self.get_versionedfiles()
        files.check(progress_bar=pb)

    def test_check_with_keys_becomes_generator(self):
        files = self.get_versionedfiles()
        self.get_diamond_files(files)
        keys = files.keys()
        entries = files.check(keys=keys)
        seen = set()
        self.capture_stream(files, entries, seen.add, files.get_parent_map(keys), require_fulltext=True)
        self.assertEqual(set(keys), seen)

    def test_clear_cache(self):
        files = self.get_versionedfiles()
        files.clear_cache()

    def test_construct(self):
        """Each parameterised test can be constructed on a transport."""
        files = self.get_versionedfiles()

    def get_diamond_files(self, files, trailing_eol=True, left_only=False, nokeys=False):
        return get_diamond_files(files, self.key_length, trailing_eol=trailing_eol, nograph=not self.graph, left_only=left_only, nokeys=nokeys)

    def _add_content_nostoresha(self, add_lines):
        """When nostore_sha is supplied using old content raises."""
        vf = self.get_versionedfiles()
        empty_text = (b'a', [])
        sample_text_nl = (b'b', [b'foo\n', b'bar\n'])
        sample_text_no_nl = (b'c', [b'foo\n', b'bar'])
        shas = []
        for version, lines in (empty_text, sample_text_nl, sample_text_no_nl):
            if add_lines:
                sha, _, _ = vf.add_lines(self.get_simple_key(version), [], lines)
            else:
                sha, _, _ = vf.add_lines(self.get_simple_key(version), [], lines)
            shas.append(sha)
        for sha, (version, lines) in zip(shas, (empty_text, sample_text_nl, sample_text_no_nl)):
            new_key = self.get_simple_key(version + b'2')
            self.assertRaises(ExistingContent, vf.add_lines, new_key, [], lines, nostore_sha=sha)
            self.assertRaises(ExistingContent, vf.add_lines, new_key, [], lines, nostore_sha=sha)
            record = next(vf.get_record_stream([new_key], 'unordered', True))
            self.assertEqual('absent', record.storage_kind)

    def test_add_lines_nostoresha(self):
        self._add_content_nostoresha(add_lines=True)

    def test_add_lines_return(self):
        files = self.get_versionedfiles()
        adds = self.get_diamond_files(files)
        results = []
        for add in adds:
            self.assertEqual(3, len(add))
            results.append(add[:2])
        if self.key_length == 1:
            self.assertEqual([(b'00e364d235126be43292ab09cb4686cf703ddc17', 7), (b'51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44', 5), (b'a8478686da38e370e32e42e8a0c220e33ee9132f', 10), (b'9ef09dfa9d86780bdec9219a22560c6ece8e0ef1', 11), (b'ed8bce375198ea62444dc71952b22cfc2b09226d', 23)], results)
        elif self.key_length == 2:
            self.assertEqual([(b'00e364d235126be43292ab09cb4686cf703ddc17', 7), (b'00e364d235126be43292ab09cb4686cf703ddc17', 7), (b'51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44', 5), (b'51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44', 5), (b'a8478686da38e370e32e42e8a0c220e33ee9132f', 10), (b'a8478686da38e370e32e42e8a0c220e33ee9132f', 10), (b'9ef09dfa9d86780bdec9219a22560c6ece8e0ef1', 11), (b'9ef09dfa9d86780bdec9219a22560c6ece8e0ef1', 11), (b'ed8bce375198ea62444dc71952b22cfc2b09226d', 23), (b'ed8bce375198ea62444dc71952b22cfc2b09226d', 23)], results)

    def test_add_lines_no_key_generates_chk_key(self):
        files = self.get_versionedfiles()
        adds = self.get_diamond_files(files, nokeys=True)
        results = []
        for add in adds:
            self.assertEqual(3, len(add))
            results.append(add[:2])
        if self.key_length == 1:
            self.assertEqual([(b'00e364d235126be43292ab09cb4686cf703ddc17', 7), (b'51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44', 5), (b'a8478686da38e370e32e42e8a0c220e33ee9132f', 10), (b'9ef09dfa9d86780bdec9219a22560c6ece8e0ef1', 11), (b'ed8bce375198ea62444dc71952b22cfc2b09226d', 23)], results)
            self.assertEqual({(b'sha1:00e364d235126be43292ab09cb4686cf703ddc17',), (b'sha1:51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44',), (b'sha1:9ef09dfa9d86780bdec9219a22560c6ece8e0ef1',), (b'sha1:a8478686da38e370e32e42e8a0c220e33ee9132f',), (b'sha1:ed8bce375198ea62444dc71952b22cfc2b09226d',)}, files.keys())
        elif self.key_length == 2:
            self.assertEqual([(b'00e364d235126be43292ab09cb4686cf703ddc17', 7), (b'00e364d235126be43292ab09cb4686cf703ddc17', 7), (b'51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44', 5), (b'51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44', 5), (b'a8478686da38e370e32e42e8a0c220e33ee9132f', 10), (b'a8478686da38e370e32e42e8a0c220e33ee9132f', 10), (b'9ef09dfa9d86780bdec9219a22560c6ece8e0ef1', 11), (b'9ef09dfa9d86780bdec9219a22560c6ece8e0ef1', 11), (b'ed8bce375198ea62444dc71952b22cfc2b09226d', 23), (b'ed8bce375198ea62444dc71952b22cfc2b09226d', 23)], results)
            self.assertEqual({(b'FileA', b'sha1:00e364d235126be43292ab09cb4686cf703ddc17'), (b'FileA', b'sha1:51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44'), (b'FileA', b'sha1:9ef09dfa9d86780bdec9219a22560c6ece8e0ef1'), (b'FileA', b'sha1:a8478686da38e370e32e42e8a0c220e33ee9132f'), (b'FileA', b'sha1:ed8bce375198ea62444dc71952b22cfc2b09226d'), (b'FileB', b'sha1:00e364d235126be43292ab09cb4686cf703ddc17'), (b'FileB', b'sha1:51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44'), (b'FileB', b'sha1:9ef09dfa9d86780bdec9219a22560c6ece8e0ef1'), (b'FileB', b'sha1:a8478686da38e370e32e42e8a0c220e33ee9132f'), (b'FileB', b'sha1:ed8bce375198ea62444dc71952b22cfc2b09226d')}, files.keys())

    def test_empty_lines(self):
        """Empty files can be stored."""
        f = self.get_versionedfiles()
        key_a = self.get_simple_key(b'a')
        f.add_lines(key_a, [], [])
        self.assertEqual(b'', next(f.get_record_stream([key_a], 'unordered', True)).get_bytes_as('fulltext'))
        key_b = self.get_simple_key(b'b')
        f.add_lines(key_b, self.get_parents([key_a]), [])
        self.assertEqual(b'', next(f.get_record_stream([key_b], 'unordered', True)).get_bytes_as('fulltext'))

    def test_newline_only(self):
        f = self.get_versionedfiles()
        key_a = self.get_simple_key(b'a')
        f.add_lines(key_a, [], [b'\n'])
        self.assertEqual(b'\n', next(f.get_record_stream([key_a], 'unordered', True)).get_bytes_as('fulltext'))
        key_b = self.get_simple_key(b'b')
        f.add_lines(key_b, self.get_parents([key_a]), [b'\n'])
        self.assertEqual(b'\n', next(f.get_record_stream([key_b], 'unordered', True)).get_bytes_as('fulltext'))

    def test_get_known_graph_ancestry(self):
        f = self.get_versionedfiles()
        if not self.graph:
            raise TestNotApplicable('ancestry info only relevant with graph.')
        key_a = self.get_simple_key(b'a')
        key_b = self.get_simple_key(b'b')
        key_c = self.get_simple_key(b'c')
        f.add_lines(key_a, [], [b'\n'])
        f.add_lines(key_b, [key_a], [b'\n'])
        f.add_lines(key_c, [key_a, key_b], [b'\n'])
        kg = f.get_known_graph_ancestry([key_c])
        self.assertIsInstance(kg, _mod_graph.KnownGraph)
        self.assertEqual([key_a, key_b, key_c], list(kg.topo_sort()))

    def test_known_graph_with_fallbacks(self):
        f = self.get_versionedfiles('files')
        if not self.graph:
            raise TestNotApplicable('ancestry info only relevant with graph.')
        if getattr(f, 'add_fallback_versioned_files', None) is None:
            raise TestNotApplicable("%s doesn't support fallbacks" % (f.__class__.__name__,))
        key_a = self.get_simple_key(b'a')
        key_b = self.get_simple_key(b'b')
        key_c = self.get_simple_key(b'c')
        g = self.get_versionedfiles('fallback')
        g.add_lines(key_a, [], [b'\n'])
        f.add_fallback_versioned_files(g)
        f.add_lines(key_b, [key_a], [b'\n'])
        f.add_lines(key_c, [key_a, key_b], [b'\n'])
        kg = f.get_known_graph_ancestry([key_c])
        self.assertEqual([key_a, key_b, key_c], list(kg.topo_sort()))

    def test_get_record_stream_empty(self):
        """An empty stream can be requested without error."""
        f = self.get_versionedfiles()
        entries = f.get_record_stream([], 'unordered', False)
        self.assertEqual([], list(entries))

    def assertValidStorageKind(self, storage_kind):
        """Assert that storage_kind is a valid storage_kind."""
        self.assertSubset([storage_kind], ['mpdiff', 'knit-annotated-ft', 'knit-annotated-delta', 'knit-ft', 'knit-delta', 'chunked', 'fulltext', 'knit-annotated-ft-gz', 'knit-annotated-delta-gz', 'knit-ft-gz', 'knit-delta-gz', 'knit-delta-closure', 'knit-delta-closure-ref', 'groupcompress-block', 'groupcompress-block-ref'])

    def capture_stream(self, f, entries, on_seen, parents, require_fulltext=False):
        """Capture a stream for testing."""
        for factory in entries:
            on_seen(factory.key)
            self.assertValidStorageKind(factory.storage_kind)
            if factory.sha1 is not None:
                self.assertEqual(f.get_sha1s([factory.key])[factory.key], factory.sha1)
            self.assertEqual(parents[factory.key], factory.parents)
            self.assertIsInstance(factory.get_bytes_as(factory.storage_kind), bytes)
            if require_fulltext:
                factory.get_bytes_as('fulltext')

    def test_get_record_stream_interface(self):
        """each item in a stream has to provide a regular interface."""
        files = self.get_versionedfiles()
        self.get_diamond_files(files)
        keys, _ = self.get_keys_and_sort_order()
        parent_map = files.get_parent_map(keys)
        entries = files.get_record_stream(keys, 'unordered', False)
        seen = set()
        self.capture_stream(files, entries, seen.add, parent_map)
        self.assertEqual(set(keys), seen)

    def get_keys_and_sort_order(self):
        """Get diamond test keys list, and their sort ordering."""
        if self.key_length == 1:
            keys = [(b'merged',), (b'left',), (b'right',), (b'base',)]
            sort_order = {(b'merged',): 2, (b'left',): 1, (b'right',): 1, (b'base',): 0}
        else:
            keys = [(b'FileA', b'merged'), (b'FileA', b'left'), (b'FileA', b'right'), (b'FileA', b'base'), (b'FileB', b'merged'), (b'FileB', b'left'), (b'FileB', b'right'), (b'FileB', b'base')]
            sort_order = {(b'FileA', b'merged'): 2, (b'FileA', b'left'): 1, (b'FileA', b'right'): 1, (b'FileA', b'base'): 0, (b'FileB', b'merged'): 2, (b'FileB', b'left'): 1, (b'FileB', b'right'): 1, (b'FileB', b'base'): 0}
        return (keys, sort_order)

    def get_keys_and_groupcompress_sort_order(self):
        """Get diamond test keys list, and their groupcompress sort ordering."""
        if self.key_length == 1:
            keys = [(b'merged',), (b'left',), (b'right',), (b'base',)]
            sort_order = {(b'merged',): 0, (b'left',): 1, (b'right',): 1, (b'base',): 2}
        else:
            keys = [(b'FileA', b'merged'), (b'FileA', b'left'), (b'FileA', b'right'), (b'FileA', b'base'), (b'FileB', b'merged'), (b'FileB', b'left'), (b'FileB', b'right'), (b'FileB', b'base')]
            sort_order = {(b'FileA', b'merged'): 0, (b'FileA', b'left'): 1, (b'FileA', b'right'): 1, (b'FileA', b'base'): 2, (b'FileB', b'merged'): 3, (b'FileB', b'left'): 4, (b'FileB', b'right'): 4, (b'FileB', b'base'): 5}
        return (keys, sort_order)

    def test_get_record_stream_interface_ordered(self):
        """each item in a stream has to provide a regular interface."""
        files = self.get_versionedfiles()
        self.get_diamond_files(files)
        keys, sort_order = self.get_keys_and_sort_order()
        parent_map = files.get_parent_map(keys)
        entries = files.get_record_stream(keys, 'topological', False)
        seen = []
        self.capture_stream(files, entries, seen.append, parent_map)
        self.assertStreamOrder(sort_order, seen, keys)

    def test_get_record_stream_interface_ordered_with_delta_closure(self):
        """each item must be accessible as a fulltext."""
        files = self.get_versionedfiles()
        self.get_diamond_files(files)
        keys, sort_order = self.get_keys_and_sort_order()
        parent_map = files.get_parent_map(keys)
        entries = files.get_record_stream(keys, 'topological', True)
        seen = []
        for factory in entries:
            seen.append(factory.key)
            self.assertValidStorageKind(factory.storage_kind)
            self.assertSubset([factory.sha1], [None, files.get_sha1s([factory.key])[factory.key]])
            self.assertEqual(parent_map[factory.key], factory.parents)
            ft_bytes = factory.get_bytes_as('fulltext')
            self.assertIsInstance(ft_bytes, bytes)
            chunked_bytes = factory.get_bytes_as('chunked')
            self.assertEqualDiff(ft_bytes, b''.join(chunked_bytes))
            chunked_bytes = factory.iter_bytes_as('chunked')
            self.assertEqualDiff(ft_bytes, b''.join(chunked_bytes))
        self.assertStreamOrder(sort_order, seen, keys)

    def test_get_record_stream_interface_groupcompress(self):
        """each item in a stream has to provide a regular interface."""
        files = self.get_versionedfiles()
        self.get_diamond_files(files)
        keys, sort_order = self.get_keys_and_groupcompress_sort_order()
        parent_map = files.get_parent_map(keys)
        entries = files.get_record_stream(keys, 'groupcompress', False)
        seen = []
        self.capture_stream(files, entries, seen.append, parent_map)
        self.assertStreamOrder(sort_order, seen, keys)

    def assertStreamOrder(self, sort_order, seen, keys):
        self.assertEqual(len(set(seen)), len(keys))
        if self.key_length == 1:
            lows = {(): 0}
        else:
            lows = {(b'FileA',): 0, (b'FileB',): 0}
        if not self.graph:
            self.assertEqual(set(keys), set(seen))
        else:
            for key in seen:
                sort_pos = sort_order[key]
                self.assertTrue(sort_pos >= lows[key[:-1]], 'Out of order in sorted stream: {!r}, {!r}'.format(key, seen))
                lows[key[:-1]] = sort_pos

    def test_get_record_stream_unknown_storage_kind_raises(self):
        """Asking for a storage kind that the stream cannot supply raises."""
        files = self.get_versionedfiles()
        self.get_diamond_files(files)
        if self.key_length == 1:
            keys = [(b'merged',), (b'left',), (b'right',), (b'base',)]
        else:
            keys = [(b'FileA', b'merged'), (b'FileA', b'left'), (b'FileA', b'right'), (b'FileA', b'base'), (b'FileB', b'merged'), (b'FileB', b'left'), (b'FileB', b'right'), (b'FileB', b'base')]
        parent_map = files.get_parent_map(keys)
        entries = files.get_record_stream(keys, 'unordered', False)
        seen = set()
        for factory in entries:
            seen.add(factory.key)
            self.assertValidStorageKind(factory.storage_kind)
            if factory.sha1 is not None:
                self.assertEqual(files.get_sha1s([factory.key])[factory.key], factory.sha1)
            self.assertEqual(parent_map[factory.key], factory.parents)
            self.assertRaises(UnavailableRepresentation, factory.get_bytes_as, 'mpdiff')
            self.assertIsInstance(factory.get_bytes_as(factory.storage_kind), bytes)
        self.assertEqual(set(keys), seen)

    def test_get_record_stream_missing_records_are_absent(self):
        files = self.get_versionedfiles()
        self.get_diamond_files(files)
        if self.key_length == 1:
            keys = [(b'merged',), (b'left',), (b'right',), (b'absent',), (b'base',)]
        else:
            keys = [(b'FileA', b'merged'), (b'FileA', b'left'), (b'FileA', b'right'), (b'FileA', b'absent'), (b'FileA', b'base'), (b'FileB', b'merged'), (b'FileB', b'left'), (b'FileB', b'right'), (b'FileB', b'absent'), (b'FileB', b'base'), (b'absent', b'absent')]
        parent_map = files.get_parent_map(keys)
        entries = files.get_record_stream(keys, 'unordered', False)
        self.assertAbsentRecord(files, keys, parent_map, entries)
        entries = files.get_record_stream(keys, 'topological', False)
        self.assertAbsentRecord(files, keys, parent_map, entries)

    def assertRecordHasContent(self, record, bytes):
        """Assert that record has the bytes bytes."""
        self.assertEqual(bytes, record.get_bytes_as('fulltext'))
        self.assertEqual(bytes, b''.join(record.get_bytes_as('chunked')))

    def test_get_record_stream_native_formats_are_wire_ready_one_ft(self):
        files = self.get_versionedfiles()
        key = self.get_simple_key(b'foo')
        files.add_lines(key, (), [b'my text\n', b'content'])
        stream = files.get_record_stream([key], 'unordered', False)
        record = next(stream)
        if record.storage_kind in ('chunked', 'fulltext'):
            self.assertRecordHasContent(record, b'my text\ncontent')
        else:
            bytes = [record.get_bytes_as(record.storage_kind)]
            network_stream = versionedfile.NetworkRecordStream(bytes).read()
            source_record = record
            records = []
            for record in network_stream:
                records.append(record)
                self.assertEqual(source_record.storage_kind, record.storage_kind)
                self.assertEqual(source_record.parents, record.parents)
                self.assertEqual(source_record.get_bytes_as(source_record.storage_kind), record.get_bytes_as(record.storage_kind))
            self.assertEqual(1, len(records))

    def assertStreamMetaEqual(self, records, expected, stream):
        """Assert that streams expected and stream have the same records.

        :param records: A list to collect the seen records.
        :return: A generator of the records in stream.
        """
        for record, ref_record in zip(stream, expected):
            records.append(record)
            self.assertEqual(ref_record.key, record.key)
            self.assertEqual(ref_record.storage_kind, record.storage_kind)
            self.assertEqual(ref_record.parents, record.parents)
            yield record

    def stream_to_bytes_or_skip_counter(self, skipped_records, full_texts, stream):
        """Convert a stream to a bytes iterator.

        :param skipped_records: A list with one element to increment when a
            record is skipped.
        :param full_texts: A dict from key->fulltext representation, for
            checking chunked or fulltext stored records.
        :param stream: A record_stream.
        :return: An iterator over the bytes of each record.
        """
        for record in stream:
            if record.storage_kind in ('chunked', 'fulltext'):
                skipped_records[0] += 1
                self.assertRecordHasContent(record, full_texts[record.key])
            else:
                yield record.get_bytes_as(record.storage_kind)

    def test_get_record_stream_native_formats_are_wire_ready_ft_delta(self):
        files = self.get_versionedfiles()
        target_files = self.get_versionedfiles('target')
        key = self.get_simple_key(b'ft')
        key_delta = self.get_simple_key(b'delta')
        files.add_lines(key, (), [b'my text\n', b'content'])
        if self.graph:
            delta_parents = (key,)
        else:
            delta_parents = ()
        files.add_lines(key_delta, delta_parents, [b'different\n', b'content\n'])
        local = files.get_record_stream([key, key_delta], 'unordered', False)
        ref = files.get_record_stream([key, key_delta], 'unordered', False)
        skipped_records = [0]
        full_texts = {key: b'my text\ncontent', key_delta: b'different\ncontent\n'}
        byte_stream = self.stream_to_bytes_or_skip_counter(skipped_records, full_texts, local)
        network_stream = versionedfile.NetworkRecordStream(byte_stream).read()
        records = []
        target_files.insert_record_stream(self.assertStreamMetaEqual(records, ref, network_stream))
        self.assertEqual(2, len(records) + skipped_records[0])
        if len(records):
            self.assertIdenticalVersionedFile(files, target_files)

    def test_get_record_stream_native_formats_are_wire_ready_delta(self):
        files = self.get_versionedfiles()
        target_files = self.get_versionedfiles('target')
        key = self.get_simple_key(b'ft')
        key_delta = self.get_simple_key(b'delta')
        files.add_lines(key, (), [b'my text\n', b'content'])
        if self.graph:
            delta_parents = (key,)
        else:
            delta_parents = ()
        files.add_lines(key_delta, delta_parents, [b'different\n', b'content\n'])
        target_files.insert_record_stream(files.get_record_stream([key], 'unordered', False))
        local = files.get_record_stream([key_delta], 'unordered', False)
        ref = files.get_record_stream([key_delta], 'unordered', False)
        skipped_records = [0]
        full_texts = {key_delta: b'different\ncontent\n'}
        byte_stream = self.stream_to_bytes_or_skip_counter(skipped_records, full_texts, local)
        network_stream = versionedfile.NetworkRecordStream(byte_stream).read()
        records = []
        target_files.insert_record_stream(self.assertStreamMetaEqual(records, ref, network_stream))
        self.assertEqual(1, len(records) + skipped_records[0])
        if len(records):
            self.assertIdenticalVersionedFile(files, target_files)

    def test_get_record_stream_wire_ready_delta_closure_included(self):
        files = self.get_versionedfiles()
        key = self.get_simple_key(b'ft')
        key_delta = self.get_simple_key(b'delta')
        files.add_lines(key, (), [b'my text\n', b'content'])
        if self.graph:
            delta_parents = (key,)
        else:
            delta_parents = ()
        files.add_lines(key_delta, delta_parents, [b'different\n', b'content\n'])
        local = files.get_record_stream([key_delta], 'unordered', True)
        ref = files.get_record_stream([key_delta], 'unordered', True)
        skipped_records = [0]
        full_texts = {key_delta: b'different\ncontent\n'}
        byte_stream = self.stream_to_bytes_or_skip_counter(skipped_records, full_texts, local)
        network_stream = versionedfile.NetworkRecordStream(byte_stream).read()
        records = []
        for record in self.assertStreamMetaEqual(records, ref, network_stream):
            self.assertRecordHasContent(record, full_texts[record.key])
        self.assertEqual(1, len(records) + skipped_records[0])

    def assertAbsentRecord(self, files, keys, parents, entries):
        """Helper for test_get_record_stream_missing_records_are_absent."""
        seen = set()
        for factory in entries:
            seen.add(factory.key)
            if factory.key[-1] == b'absent':
                self.assertEqual('absent', factory.storage_kind)
                self.assertEqual(None, factory.sha1)
                self.assertEqual(None, factory.parents)
            else:
                self.assertValidStorageKind(factory.storage_kind)
                if factory.sha1 is not None:
                    sha1 = files.get_sha1s([factory.key])[factory.key]
                    self.assertEqual(sha1, factory.sha1)
                self.assertEqual(parents[factory.key], factory.parents)
                self.assertIsInstance(factory.get_bytes_as(factory.storage_kind), bytes)
        self.assertEqual(set(keys), seen)

    def test_filter_absent_records(self):
        """Requested missing records can be filter trivially."""
        files = self.get_versionedfiles()
        self.get_diamond_files(files)
        keys, _ = self.get_keys_and_sort_order()
        parent_map = files.get_parent_map(keys)
        present_keys = list(keys)
        if self.key_length == 1:
            keys.insert(2, (b'extra',))
        else:
            keys.insert(2, (b'extra', b'extra'))
        entries = files.get_record_stream(keys, 'unordered', False)
        seen = set()
        self.capture_stream(files, versionedfile.filter_absent(entries), seen.add, parent_map)
        self.assertEqual(set(present_keys), seen)

    def get_mapper(self):
        """Get a mapper suitable for the key length of the test interface."""
        if self.key_length == 1:
            return ConstantMapper('source')
        else:
            return HashEscapedPrefixMapper()

    def get_parents(self, parents):
        """Get parents, taking self.graph into consideration."""
        if self.graph:
            return parents
        else:
            return None

    def test_get_annotator(self):
        files = self.get_versionedfiles()
        self.get_diamond_files(files)
        origin_key = self.get_simple_key(b'origin')
        base_key = self.get_simple_key(b'base')
        left_key = self.get_simple_key(b'left')
        right_key = self.get_simple_key(b'right')
        merged_key = self.get_simple_key(b'merged')
        origins, lines = files.get_annotator().annotate(origin_key)
        self.assertEqual([(origin_key,)], origins)
        self.assertEqual([b'origin\n'], lines)
        origins, lines = files.get_annotator().annotate(base_key)
        self.assertEqual([(base_key,)], origins)
        origins, lines = files.get_annotator().annotate(merged_key)
        if self.graph:
            self.assertEqual([(base_key,), (left_key,), (right_key,), (merged_key,)], origins)
        else:
            self.assertEqual([(merged_key,), (merged_key,), (merged_key,), (merged_key,)], origins)
        self.assertRaises(RevisionNotPresent, files.get_annotator().annotate, self.get_simple_key(b'missing-key'))

    def test_get_parent_map(self):
        files = self.get_versionedfiles()
        if self.key_length == 1:
            parent_details = [((b'r0',), self.get_parents(())), ((b'r1',), self.get_parents(((b'r0',),))), ((b'r2',), self.get_parents(())), ((b'r3',), self.get_parents(())), ((b'm',), self.get_parents(((b'r0',), (b'r1',), (b'r2',), (b'r3',))))]
        else:
            parent_details = [((b'FileA', b'r0'), self.get_parents(())), ((b'FileA', b'r1'), self.get_parents(((b'FileA', b'r0'),))), ((b'FileA', b'r2'), self.get_parents(())), ((b'FileA', b'r3'), self.get_parents(())), ((b'FileA', b'm'), self.get_parents(((b'FileA', b'r0'), (b'FileA', b'r1'), (b'FileA', b'r2'), (b'FileA', b'r3'))))]
        for key, parents in parent_details:
            files.add_lines(key, parents, [])
            self.assertEqual({key: parents}, files.get_parent_map([key]))
        self.assertEqual({}, files.get_parent_map([]))
        all_parents = dict(parent_details)
        self.assertEqual(all_parents, files.get_parent_map(all_parents.keys()))
        keys = list(all_parents.keys())
        if self.key_length == 1:
            keys.insert(1, (b'missing',))
        else:
            keys.insert(1, (b'missing', b'missing'))
        self.assertEqual(all_parents, files.get_parent_map(keys))

    def test_get_sha1s(self):
        files = self.get_versionedfiles()
        self.get_diamond_files(files)
        if self.key_length == 1:
            keys = [(b'base',), (b'origin',), (b'left',), (b'merged',), (b'right',)]
        else:
            keys = [(b'FileA', b'base'), (b'FileB', b'origin'), (b'FileA', b'left'), (b'FileA', b'merged'), (b'FileB', b'right')]
        self.assertEqual({keys[0]: b'51c64a6f4fc375daf0d24aafbabe4d91b6f4bb44', keys[1]: b'00e364d235126be43292ab09cb4686cf703ddc17', keys[2]: b'a8478686da38e370e32e42e8a0c220e33ee9132f', keys[3]: b'ed8bce375198ea62444dc71952b22cfc2b09226d', keys[4]: b'9ef09dfa9d86780bdec9219a22560c6ece8e0ef1'}, files.get_sha1s(keys))

    def test_insert_record_stream_empty(self):
        """Inserting an empty record stream should work."""
        files = self.get_versionedfiles()
        files.insert_record_stream([])

    def assertIdenticalVersionedFile(self, expected, actual):
        """Assert that left and right have the same contents."""
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        actual_parents = actual.get_parent_map(actual.keys())
        if self.graph:
            self.assertEqual(actual_parents, expected.get_parent_map(expected.keys()))
        else:
            for key, parents in actual_parents.items():
                self.assertEqual(None, parents)
        for key in actual.keys():
            actual_text = next(actual.get_record_stream([key], 'unordered', True)).get_bytes_as('fulltext')
            expected_text = next(expected.get_record_stream([key], 'unordered', True)).get_bytes_as('fulltext')
            self.assertEqual(actual_text, expected_text)

    def test_insert_record_stream_fulltexts(self):
        """Any file should accept a stream of fulltexts."""
        files = self.get_versionedfiles()
        mapper = self.get_mapper()
        source_transport = self.get_transport('source')
        source_transport.mkdir('.')
        source = make_versioned_files_factory(WeaveFile, mapper)(source_transport)
        self.get_diamond_files(source, trailing_eol=False)
        stream = source.get_record_stream(source.keys(), 'topological', False)
        files.insert_record_stream(stream)
        self.assertIdenticalVersionedFile(source, files)

    def test_insert_record_stream_fulltexts_noeol(self):
        """Any file should accept a stream of fulltexts."""
        files = self.get_versionedfiles()
        mapper = self.get_mapper()
        source_transport = self.get_transport('source')
        source_transport.mkdir('.')
        source = make_versioned_files_factory(WeaveFile, mapper)(source_transport)
        self.get_diamond_files(source, trailing_eol=False)
        stream = source.get_record_stream(source.keys(), 'topological', False)
        files.insert_record_stream(stream)
        self.assertIdenticalVersionedFile(source, files)

    def test_insert_record_stream_annotated_knits(self):
        """Any file should accept a stream from plain knits."""
        files = self.get_versionedfiles()
        mapper = self.get_mapper()
        source_transport = self.get_transport('source')
        source_transport.mkdir('.')
        source = make_file_factory(True, mapper)(source_transport)
        self.get_diamond_files(source)
        stream = source.get_record_stream(source.keys(), 'topological', False)
        files.insert_record_stream(stream)
        self.assertIdenticalVersionedFile(source, files)

    def test_insert_record_stream_annotated_knits_noeol(self):
        """Any file should accept a stream from plain knits."""
        files = self.get_versionedfiles()
        mapper = self.get_mapper()
        source_transport = self.get_transport('source')
        source_transport.mkdir('.')
        source = make_file_factory(True, mapper)(source_transport)
        self.get_diamond_files(source, trailing_eol=False)
        stream = source.get_record_stream(source.keys(), 'topological', False)
        files.insert_record_stream(stream)
        self.assertIdenticalVersionedFile(source, files)

    def test_insert_record_stream_plain_knits(self):
        """Any file should accept a stream from plain knits."""
        files = self.get_versionedfiles()
        mapper = self.get_mapper()
        source_transport = self.get_transport('source')
        source_transport.mkdir('.')
        source = make_file_factory(False, mapper)(source_transport)
        self.get_diamond_files(source)
        stream = source.get_record_stream(source.keys(), 'topological', False)
        files.insert_record_stream(stream)
        self.assertIdenticalVersionedFile(source, files)

    def test_insert_record_stream_plain_knits_noeol(self):
        """Any file should accept a stream from plain knits."""
        files = self.get_versionedfiles()
        mapper = self.get_mapper()
        source_transport = self.get_transport('source')
        source_transport.mkdir('.')
        source = make_file_factory(False, mapper)(source_transport)
        self.get_diamond_files(source, trailing_eol=False)
        stream = source.get_record_stream(source.keys(), 'topological', False)
        files.insert_record_stream(stream)
        self.assertIdenticalVersionedFile(source, files)

    def test_insert_record_stream_existing_keys(self):
        """Inserting keys already in a file should not error."""
        files = self.get_versionedfiles()
        source = self.get_versionedfiles('source')
        self.get_diamond_files(source)
        self.get_diamond_files(files, left_only=True)
        stream = source.get_record_stream(source.keys(), 'topological', False)
        files.insert_record_stream(stream)
        self.assertIdenticalVersionedFile(source, files)

    def test_insert_record_stream_missing_keys(self):
        """Inserting a stream with absent keys should raise an error."""
        files = self.get_versionedfiles()
        source = self.get_versionedfiles('source')
        stream = source.get_record_stream([(b'missing',) * self.key_length], 'topological', False)
        self.assertRaises(errors.RevisionNotPresent, files.insert_record_stream, stream)

    def test_insert_record_stream_out_of_order(self):
        """An out of order stream can either error or work."""
        files = self.get_versionedfiles()
        source = self.get_versionedfiles('source')
        self.get_diamond_files(source)
        if self.key_length == 1:
            origin_keys = [(b'origin',)]
            end_keys = [(b'merged',), (b'left',)]
            start_keys = [(b'right',), (b'base',)]
        else:
            origin_keys = [(b'FileA', b'origin'), (b'FileB', b'origin')]
            end_keys = [(b'FileA', b'merged'), (b'FileA', b'left'), (b'FileB', b'merged'), (b'FileB', b'left')]
            start_keys = [(b'FileA', b'right'), (b'FileA', b'base'), (b'FileB', b'right'), (b'FileB', b'base')]
        origin_entries = source.get_record_stream(origin_keys, 'unordered', False)
        end_entries = source.get_record_stream(end_keys, 'topological', False)
        start_entries = source.get_record_stream(start_keys, 'topological', False)
        entries = itertools.chain(origin_entries, end_entries, start_entries)
        try:
            files.insert_record_stream(entries)
        except RevisionNotPresent:
            files.check()
        else:
            self.assertIdenticalVersionedFile(source, files)

    def test_insert_record_stream_long_parent_chain_out_of_order(self):
        """An out of order stream can either error or work."""
        if not self.graph:
            raise TestNotApplicable('ancestry info only relevant with graph.')
        source = self.get_versionedfiles('source')
        parents = ()
        keys = []
        content = [b'same same %d\n' % n for n in range(500)]
        letters = b'abcdefghijklmnopqrstuvwxyz'
        for i in range(len(letters)):
            letter = letters[i:i + 1]
            key = (b'key-' + letter,)
            if self.key_length == 2:
                key = (b'prefix',) + key
            content.append(b'content for ' + letter + b'\n')
            source.add_lines(key, parents, content)
            keys.append(key)
            parents = (key,)
        streams = []
        for key in reversed(keys):
            streams.append(source.get_record_stream([key], 'unordered', False))
        deltas = itertools.chain.from_iterable(streams[:-1])
        files = self.get_versionedfiles()
        try:
            files.insert_record_stream(deltas)
        except RevisionNotPresent:
            files.check()
        else:
            missing = set(files.get_missing_compression_parent_keys())
            missing.discard(keys[0])
            self.assertEqual(set(), missing)

    def get_knit_delta_source(self):
        """Get a source that can produce a stream with knit delta records,
        regardless of this test's scenario.
        """
        mapper = self.get_mapper()
        source_transport = self.get_transport('source')
        source_transport.mkdir('.')
        source = make_file_factory(False, mapper)(source_transport)
        get_diamond_files(source, self.key_length, trailing_eol=True, nograph=False, left_only=False)
        return source

    def test_insert_record_stream_delta_missing_basis_no_corruption(self):
        """Insertion where a needed basis is not included notifies the caller
        of the missing basis.  In the meantime a record missing its basis is
        not added.
        """
        source = self.get_knit_delta_source()
        keys = [self.get_simple_key(b'origin'), self.get_simple_key(b'merged')]
        entries = source.get_record_stream(keys, 'unordered', False)
        files = self.get_versionedfiles()
        if self.support_partial_insertion:
            self.assertEqual([], list(files.get_missing_compression_parent_keys()))
            files.insert_record_stream(entries)
            missing_bases = files.get_missing_compression_parent_keys()
            self.assertEqual({self.get_simple_key(b'left')}, set(missing_bases))
            self.assertEqual(set(keys), set(files.get_parent_map(keys)))
        else:
            self.assertRaises(errors.RevisionNotPresent, files.insert_record_stream, entries)
            files.check()

    def test_insert_record_stream_delta_missing_basis_can_be_added_later(self):
        """Insertion where a needed basis is not included notifies the caller
        of the missing basis.  That basis can be added in a second
        insert_record_stream call that does not need to repeat records present
        in the previous stream.  The record(s) that required that basis are
        fully inserted once their basis is no longer missing.
        """
        if not self.support_partial_insertion:
            raise TestNotApplicable('versioned file scenario does not support partial insertion')
        source = self.get_knit_delta_source()
        entries = source.get_record_stream([self.get_simple_key(b'origin'), self.get_simple_key(b'merged')], 'unordered', False)
        files = self.get_versionedfiles()
        files.insert_record_stream(entries)
        missing_bases = files.get_missing_compression_parent_keys()
        self.assertEqual({self.get_simple_key(b'left')}, set(missing_bases))
        merged_key = self.get_simple_key(b'merged')
        self.assertEqual([merged_key], list(files.get_parent_map([merged_key]).keys()))
        missing_entries = source.get_record_stream(missing_bases, 'unordered', True)
        files.insert_record_stream(missing_entries)
        self.assertEqual([], list(files.get_missing_compression_parent_keys()))
        self.assertEqual([merged_key], list(files.get_parent_map([merged_key]).keys()))
        files.check()

    def test_iter_lines_added_or_present_in_keys(self):

        class InstrumentedProgress(progress.ProgressTask):

            def __init__(self):
                progress.ProgressTask.__init__(self)
                self.updates = []

            def update(self, msg=None, current=None, total=None):
                self.updates.append((msg, current, total))
        files = self.get_versionedfiles()
        files.add_lines(self.get_simple_key(b'base'), (), [b'base\n'])
        files.add_lines(self.get_simple_key(b'lancestor'), (), [b'lancestor\n'])
        files.add_lines(self.get_simple_key(b'rancestor'), self.get_parents([self.get_simple_key(b'base')]), [b'rancestor\n'])
        files.add_lines(self.get_simple_key(b'child'), self.get_parents([self.get_simple_key(b'rancestor')]), [b'base\n', b'child\n'])
        files.add_lines(self.get_simple_key(b'otherchild'), self.get_parents([self.get_simple_key(b'lancestor'), self.get_simple_key(b'base')]), [b'base\n', b'lancestor\n', b'otherchild\n'])

        def iter_with_keys(keys, expected):
            lines = {}
            progress = InstrumentedProgress()
            for line in files.iter_lines_added_or_present_in_keys(keys, pb=progress):
                lines.setdefault(line, 0)
                lines[line] += 1
            if [] != progress.updates:
                self.assertEqual(expected, progress.updates)
            return lines
        lines = iter_with_keys([self.get_simple_key(b'child'), self.get_simple_key(b'otherchild')], [('Walking content', 0, 2), ('Walking content', 1, 2), ('Walking content', 2, 2)])
        self.assertTrue(lines[b'child\n', self.get_simple_key(b'child')] > 0)
        self.assertTrue(lines[b'otherchild\n', self.get_simple_key(b'otherchild')] > 0)
        lines = iter_with_keys(files.keys(), [('Walking content', 0, 5), ('Walking content', 1, 5), ('Walking content', 2, 5), ('Walking content', 3, 5), ('Walking content', 4, 5), ('Walking content', 5, 5)])
        self.assertTrue(lines[b'base\n', self.get_simple_key(b'base')] > 0)
        self.assertTrue(lines[b'lancestor\n', self.get_simple_key(b'lancestor')] > 0)
        self.assertTrue(lines[b'rancestor\n', self.get_simple_key(b'rancestor')] > 0)
        self.assertTrue(lines[b'child\n', self.get_simple_key(b'child')] > 0)
        self.assertTrue(lines[b'otherchild\n', self.get_simple_key(b'otherchild')] > 0)

    def test_make_mpdiffs(self):
        from breezy import multiparent
        files = self.get_versionedfiles('source')
        files.add_lines(self.get_simple_key(b'base'), [], [b'line\n'])
        files.add_lines(self.get_simple_key(b'noeol'), self.get_parents([self.get_simple_key(b'base')]), [b'line'])
        files.add_lines(self.get_simple_key(b'noeolsecond'), self.get_parents([self.get_simple_key(b'noeol')]), [b'line\n', b'line'])
        files.add_lines(self.get_simple_key(b'noeolnotshared'), self.get_parents([self.get_simple_key(b'noeolsecond')]), [b'line\n', b'phone'])
        files.add_lines(self.get_simple_key(b'eol'), self.get_parents([self.get_simple_key(b'noeol')]), [b'phone\n'])
        files.add_lines(self.get_simple_key(b'eolline'), self.get_parents([self.get_simple_key(b'noeol')]), [b'line\n'])
        files.add_lines(self.get_simple_key(b'noeolbase'), [], [b'line'])
        files.add_lines(self.get_simple_key(b'eolbeforefirstparent'), self.get_parents([self.get_simple_key(b'noeolbase'), self.get_simple_key(b'noeol')]), [b'line'])
        files.add_lines(self.get_simple_key(b'noeoldup'), self.get_parents([self.get_simple_key(b'noeol')]), [b'line'])
        next_parent = self.get_simple_key(b'base')
        text_name = b'chain1-'
        text = [b'line\n']
        sha1s = {0: b'da6d3141cb4a5e6f464bf6e0518042ddc7bfd079', 1: b'45e21ea146a81ea44a821737acdb4f9791c8abe7', 2: b'e1f11570edf3e2a070052366c582837a4fe4e9fa', 3: b'26b4b8626da827088c514b8f9bbe4ebf181edda1', 4: b'e28a5510be25ba84d31121cff00956f9970ae6f6', 5: b'd63ec0ce22e11dcf65a931b69255d3ac747a318d', 6: b'2c2888d288cb5e1d98009d822fedfe6019c6a4ea', 7: b'95c14da9cafbf828e3e74a6f016d87926ba234ab', 8: b'779e9a0b28f9f832528d4b21e17e168c67697272', 9: b'1f8ff4e5c6ff78ac106fcfe6b1e8cb8740ff9a8f', 10: b'131a2ae712cf51ed62f143e3fbac3d4206c25a05', 11: b'c5a9d6f520d2515e1ec401a8f8a67e6c3c89f199', 12: b'31a2286267f24d8bedaa43355f8ad7129509ea85', 13: b'dc2a7fe80e8ec5cae920973973a8ee28b2da5e0a', 14: b'2c4b1736566b8ca6051e668de68650686a3922f2', 15: b'5912e4ecd9b0c07be4d013e7e2bdcf9323276cde', 16: b'b0d2e18d3559a00580f6b49804c23fea500feab3', 17: b'8e1d43ad72f7562d7cb8f57ee584e20eb1a69fc7', 18: b'5cf64a3459ae28efa60239e44b20312d25b253f3', 19: b'1ebed371807ba5935958ad0884595126e8c4e823', 20: b'2aa62a8b06fb3b3b892a3292a068ade69d5ee0d3', 21: b'01edc447978004f6e4e962b417a4ae1955b6fe5d', 22: b'd8d8dc49c4bf0bab401e0298bb5ad827768618bb', 23: b'c21f62b1c482862983a8ffb2b0c64b3451876e3f', 24: b'c0593fe795e00dff6b3c0fe857a074364d5f04fc', 25: b'dd1a1cf2ba9cc225c3aff729953e6364bf1d1855'}
        for depth in range(26):
            new_version = self.get_simple_key(text_name + b'%d' % depth)
            text = text + [b'line\n']
            files.add_lines(new_version, self.get_parents([next_parent]), text)
            next_parent = new_version
        next_parent = self.get_simple_key(b'base')
        text_name = b'chain2-'
        text = [b'line\n']
        for depth in range(26):
            new_version = self.get_simple_key(text_name + b'%d' % depth)
            text = text + [b'line\n']
            files.add_lines(new_version, self.get_parents([next_parent]), text)
            next_parent = new_version
        target = self.get_versionedfiles('target')
        for key in multiparent.topo_iter_keys(files, files.keys()):
            mpdiff = files.make_mpdiffs([key])[0]
            parents = files.get_parent_map([key])[key] or []
            target.add_mpdiffs([(key, parents, files.get_sha1s([key])[key], mpdiff)])
            self.assertEqualDiff(next(files.get_record_stream([key], 'unordered', True)).get_bytes_as('fulltext'), next(target.get_record_stream([key], 'unordered', True)).get_bytes_as('fulltext'))

    def test_keys(self):
        files = self.get_versionedfiles()
        self.assertEqual(set(), set(files.keys()))
        if self.key_length == 1:
            key = (b'foo',)
        else:
            key = (b'foo', b'bar')
        files.add_lines(key, (), [])
        self.assertEqual({key}, set(files.keys()))