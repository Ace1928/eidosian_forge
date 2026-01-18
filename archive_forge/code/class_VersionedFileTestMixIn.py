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
class VersionedFileTestMixIn:
    """A mixin test class for testing VersionedFiles.

    This is not an adaptor-style test at this point because
    theres no dynamic substitution of versioned file implementations,
    they are strictly controlled by their owning repositories.
    """

    def get_transaction(self):
        if not hasattr(self, '_transaction'):
            self._transaction = None
        return self._transaction

    def test_add(self):
        f = self.get_file()
        f.add_lines(b'r0', [], [b'a\n', b'b\n'])
        f.add_lines(b'r1', [b'r0'], [b'b\n', b'c\n'])

        def verify_file(f):
            versions = f.versions()
            self.assertTrue(b'r0' in versions)
            self.assertTrue(b'r1' in versions)
            self.assertEqual(f.get_lines(b'r0'), [b'a\n', b'b\n'])
            self.assertEqual(f.get_text(b'r0'), b'a\nb\n')
            self.assertEqual(f.get_lines(b'r1'), [b'b\n', b'c\n'])
            self.assertEqual(2, len(f))
            self.assertEqual(2, f.num_versions())
            self.assertRaises(RevisionNotPresent, f.add_lines, b'r2', [b'foo'], [])
            self.assertRaises(RevisionAlreadyPresent, f.add_lines, b'r1', [], [])
        verify_file(f)
        f = self.reopen_file(create=True)
        verify_file(f)

    def test_adds_with_parent_texts(self):
        f = self.get_file()
        parent_texts = {}
        _, _, parent_texts[b'r0'] = f.add_lines(b'r0', [], [b'a\n', b'b\n'])
        try:
            _, _, parent_texts[b'r1'] = f.add_lines_with_ghosts(b'r1', [b'r0', b'ghost'], [b'b\n', b'c\n'], parent_texts=parent_texts)
        except NotImplementedError:
            _, _, parent_texts[b'r1'] = f.add_lines(b'r1', [b'r0'], [b'b\n', b'c\n'], parent_texts=parent_texts)
        f.add_lines(b'r2', [b'r1'], [b'c\n', b'd\n'], parent_texts=parent_texts)
        self.assertNotEqual(None, parent_texts[b'r0'])
        self.assertNotEqual(None, parent_texts[b'r1'])

        def verify_file(f):
            versions = f.versions()
            self.assertTrue(b'r0' in versions)
            self.assertTrue(b'r1' in versions)
            self.assertTrue(b'r2' in versions)
            self.assertEqual(f.get_lines(b'r0'), [b'a\n', b'b\n'])
            self.assertEqual(f.get_lines(b'r1'), [b'b\n', b'c\n'])
            self.assertEqual(f.get_lines(b'r2'), [b'c\n', b'd\n'])
            self.assertEqual(3, f.num_versions())
            origins = f.annotate(b'r1')
            self.assertEqual(origins[0][0], b'r0')
            self.assertEqual(origins[1][0], b'r1')
            origins = f.annotate(b'r2')
            self.assertEqual(origins[0][0], b'r1')
            self.assertEqual(origins[1][0], b'r2')
        verify_file(f)
        f = self.reopen_file()
        verify_file(f)

    def test_add_unicode_content(self):
        vf = self.get_file()
        self.assertRaises(errors.BzrBadParameterUnicode, vf.add_lines, b'a', [], [b'a\n', 'b\n', b'c\n'])
        self.assertRaises((errors.BzrBadParameterUnicode, NotImplementedError), vf.add_lines_with_ghosts, b'a', [], [b'a\n', 'b\n', b'c\n'])

    def test_add_follows_left_matching_blocks(self):
        """If we change left_matching_blocks, delta changes

        Note: There are multiple correct deltas in this case, because
        we start with 1 "a" and we get 3.
        """
        vf = self.get_file()
        if isinstance(vf, WeaveFile):
            raise TestSkipped('WeaveFile ignores left_matching_blocks')
        vf.add_lines(b'1', [], [b'a\n'])
        vf.add_lines(b'2', [b'1'], [b'a\n', b'a\n', b'a\n'], left_matching_blocks=[(0, 0, 1), (1, 3, 0)])
        self.assertEqual([b'a\n', b'a\n', b'a\n'], vf.get_lines(b'2'))
        vf.add_lines(b'3', [b'1'], [b'a\n', b'a\n', b'a\n'], left_matching_blocks=[(0, 2, 1), (1, 3, 0)])
        self.assertEqual([b'a\n', b'a\n', b'a\n'], vf.get_lines(b'3'))

    def test_inline_newline_throws(self):
        vf = self.get_file()
        self.assertRaises(errors.BzrBadParameterContainsNewline, vf.add_lines, b'a', [], [b'a\n\n'])
        self.assertRaises((errors.BzrBadParameterContainsNewline, NotImplementedError), vf.add_lines_with_ghosts, b'a', [], [b'a\n\n'])
        vf.add_lines(b'a', [], [b'a\r\n'])
        try:
            vf.add_lines_with_ghosts(b'b', [], [b'a\r\n'])
        except NotImplementedError:
            pass

    def test_add_reserved(self):
        vf = self.get_file()
        self.assertRaises(errors.ReservedId, vf.add_lines, b'a:', [], [b'a\n', b'b\n', b'c\n'])

    def test_add_lines_nostoresha(self):
        """When nostore_sha is supplied using old content raises."""
        vf = self.get_file()
        empty_text = (b'a', [])
        sample_text_nl = (b'b', [b'foo\n', b'bar\n'])
        sample_text_no_nl = (b'c', [b'foo\n', b'bar'])
        shas = []
        for version, lines in (empty_text, sample_text_nl, sample_text_no_nl):
            sha, _, _ = vf.add_lines(version, [], lines)
            shas.append(sha)
        for sha, (version, lines) in zip(shas, (empty_text, sample_text_nl, sample_text_no_nl)):
            self.assertRaises(ExistingContent, vf.add_lines, version + b'2', [], lines, nostore_sha=sha)
            self.assertRaises(errors.RevisionNotPresent, vf.get_lines, version + b'2')

    def test_add_lines_with_ghosts_nostoresha(self):
        """When nostore_sha is supplied using old content raises."""
        vf = self.get_file()
        empty_text = (b'a', [])
        sample_text_nl = (b'b', [b'foo\n', b'bar\n'])
        sample_text_no_nl = (b'c', [b'foo\n', b'bar'])
        shas = []
        for version, lines in (empty_text, sample_text_nl, sample_text_no_nl):
            sha, _, _ = vf.add_lines(version, [], lines)
            shas.append(sha)
        try:
            vf.add_lines_with_ghosts(b'd', [], [])
        except NotImplementedError:
            raise TestSkipped('add_lines_with_ghosts is optional')
        for sha, (version, lines) in zip(shas, (empty_text, sample_text_nl, sample_text_no_nl)):
            self.assertRaises(ExistingContent, vf.add_lines_with_ghosts, version + b'2', [], lines, nostore_sha=sha)
            self.assertRaises(errors.RevisionNotPresent, vf.get_lines, version + b'2')

    def test_add_lines_return_value(self):
        vf = self.get_file()
        empty_text = (b'a', [])
        sample_text_nl = (b'b', [b'foo\n', b'bar\n'])
        sample_text_no_nl = (b'c', [b'foo\n', b'bar'])
        for version, lines in (empty_text, sample_text_nl, sample_text_no_nl):
            result = vf.add_lines(version, [], lines)
            self.assertEqual(3, len(result))
            self.assertEqual((osutils.sha_strings(lines), sum(map(len, lines))), result[0:2])
        lines = sample_text_nl[1]
        self.assertEqual((osutils.sha_strings(lines), sum(map(len, lines))), vf.add_lines(b'd', [b'b', b'c'], lines)[0:2])

    def test_get_reserved(self):
        vf = self.get_file()
        self.assertRaises(errors.ReservedId, vf.get_texts, [b'b:'])
        self.assertRaises(errors.ReservedId, vf.get_lines, b'b:')
        self.assertRaises(errors.ReservedId, vf.get_text, b'b:')

    def test_add_unchanged_last_line_noeol_snapshot(self):
        """Add a text with an unchanged last line with no eol should work."""
        for length in range(20):
            version_lines = {}
            vf = self.get_file('case-%d' % length)
            prefix = b'step-%d'
            parents = []
            for step in range(length):
                version = prefix % step
                lines = [b'prelude \n'] * step + [b'line']
                vf.add_lines(version, parents, lines)
                version_lines[version] = lines
                parents = [version]
            vf.add_lines(b'no-eol', parents, [b'line'])
            vf.get_texts(version_lines.keys())
            self.assertEqualDiff(b'line', vf.get_text(b'no-eol'))

    def test_get_texts_eol_variation(self):
        vf = self.get_file()
        sample_text_nl = [b'line\n']
        sample_text_no_nl = [b'line']
        versions = []
        version_lines = {}
        parents = []
        for i in range(4):
            version = b'v%d' % i
            if i % 2:
                lines = sample_text_nl
            else:
                lines = sample_text_no_nl
            vf.add_lines(version, parents, lines, left_matching_blocks=[(0, 0, 1)])
            parents = [version]
            versions.append(version)
            version_lines[version] = lines
        vf.check()
        vf.get_texts(versions)
        vf.get_texts(reversed(versions))

    def test_add_lines_with_matching_blocks_noeol_last_line(self):
        """Add a text with an unchanged last line with no eol should work."""
        from breezy import multiparent
        sha1 = '6a1d115ec7b60afb664dc14890b5af5ce3c827a4'
        vf = self.get_file('fulltext')
        vf.add_lines(b'noeol', [], [b'line'])
        vf.add_lines(b'noeol2', [b'noeol'], [b'newline\n', b'line'], left_matching_blocks=[(0, 1, 1)])
        self.assertEqualDiff(b'newline\nline', vf.get_text(b'noeol2'))
        vf = self.get_file('delta')
        vf.add_lines(b'base', [], [b'line'])
        vf.add_lines(b'noeol', [b'base'], [b'prelude\n', b'line'])
        vf.add_lines(b'noeol2', [b'noeol'], [b'newline\n', b'line'], left_matching_blocks=[(1, 1, 1)])
        self.assertEqualDiff(b'newline\nline', vf.get_text(b'noeol2'))

    def test_make_mpdiffs(self):
        from breezy import multiparent
        vf = self.get_file('foo')
        sha1s = self._setup_for_deltas(vf)
        new_vf = self.get_file('bar')
        for version in multiparent.topo_iter(vf):
            mpdiff = vf.make_mpdiffs([version])[0]
            new_vf.add_mpdiffs([(version, vf.get_parent_map([version])[version], vf.get_sha1s([version])[version], mpdiff)])
            self.assertEqualDiff(vf.get_text(version), new_vf.get_text(version))

    def test_make_mpdiffs_with_ghosts(self):
        vf = self.get_file('foo')
        try:
            vf.add_lines_with_ghosts(b'text', [b'ghost'], [b'line\n'])
        except NotImplementedError:
            return
        self.assertRaises(errors.RevisionNotPresent, vf.make_mpdiffs, [b'ghost'])

    def _setup_for_deltas(self, f):
        self.assertFalse(f.has_version('base'))
        f.add_lines(b'base', [], [b'line\n'])
        f.add_lines(b'noeol', [b'base'], [b'line'])
        f.add_lines(b'noeolsecond', [b'noeol'], [b'line\n', b'line'])
        f.add_lines(b'noeolnotshared', [b'noeolsecond'], [b'line\n', b'phone'])
        f.add_lines(b'eol', [b'noeol'], [b'phone\n'])
        f.add_lines(b'eolline', [b'noeol'], [b'line\n'])
        f.add_lines(b'noeolbase', [], [b'line'])
        f.add_lines(b'eolbeforefirstparent', [b'noeolbase', b'noeol'], [b'line'])
        f.add_lines(b'noeoldup', [b'noeol'], [b'line'])
        next_parent = b'base'
        text_name = b'chain1-'
        text = [b'line\n']
        sha1s = {0: b'da6d3141cb4a5e6f464bf6e0518042ddc7bfd079', 1: b'45e21ea146a81ea44a821737acdb4f9791c8abe7', 2: b'e1f11570edf3e2a070052366c582837a4fe4e9fa', 3: b'26b4b8626da827088c514b8f9bbe4ebf181edda1', 4: b'e28a5510be25ba84d31121cff00956f9970ae6f6', 5: b'd63ec0ce22e11dcf65a931b69255d3ac747a318d', 6: b'2c2888d288cb5e1d98009d822fedfe6019c6a4ea', 7: b'95c14da9cafbf828e3e74a6f016d87926ba234ab', 8: b'779e9a0b28f9f832528d4b21e17e168c67697272', 9: b'1f8ff4e5c6ff78ac106fcfe6b1e8cb8740ff9a8f', 10: b'131a2ae712cf51ed62f143e3fbac3d4206c25a05', 11: b'c5a9d6f520d2515e1ec401a8f8a67e6c3c89f199', 12: b'31a2286267f24d8bedaa43355f8ad7129509ea85', 13: b'dc2a7fe80e8ec5cae920973973a8ee28b2da5e0a', 14: b'2c4b1736566b8ca6051e668de68650686a3922f2', 15: b'5912e4ecd9b0c07be4d013e7e2bdcf9323276cde', 16: b'b0d2e18d3559a00580f6b49804c23fea500feab3', 17: b'8e1d43ad72f7562d7cb8f57ee584e20eb1a69fc7', 18: b'5cf64a3459ae28efa60239e44b20312d25b253f3', 19: b'1ebed371807ba5935958ad0884595126e8c4e823', 20: b'2aa62a8b06fb3b3b892a3292a068ade69d5ee0d3', 21: b'01edc447978004f6e4e962b417a4ae1955b6fe5d', 22: b'd8d8dc49c4bf0bab401e0298bb5ad827768618bb', 23: b'c21f62b1c482862983a8ffb2b0c64b3451876e3f', 24: b'c0593fe795e00dff6b3c0fe857a074364d5f04fc', 25: b'dd1a1cf2ba9cc225c3aff729953e6364bf1d1855'}
        for depth in range(26):
            new_version = text_name + b'%d' % depth
            text = text + [b'line\n']
            f.add_lines(new_version, [next_parent], text)
            next_parent = new_version
        next_parent = b'base'
        text_name = b'chain2-'
        text = [b'line\n']
        for depth in range(26):
            new_version = text_name + b'%d' % depth
            text = text + [b'line\n']
            f.add_lines(new_version, [next_parent], text)
            next_parent = new_version
        return sha1s

    def test_ancestry(self):
        f = self.get_file()
        self.assertEqual(set(), f.get_ancestry([]))
        f.add_lines(b'r0', [], [b'a\n', b'b\n'])
        f.add_lines(b'r1', [b'r0'], [b'b\n', b'c\n'])
        f.add_lines(b'r2', [b'r0'], [b'b\n', b'c\n'])
        f.add_lines(b'r3', [b'r2'], [b'b\n', b'c\n'])
        f.add_lines(b'rM', [b'r1', b'r2'], [b'b\n', b'c\n'])
        self.assertEqual(set(), f.get_ancestry([]))
        versions = f.get_ancestry([b'rM'])
        self.assertRaises(RevisionNotPresent, f.get_ancestry, [b'rM', b'rX'])
        self.assertEqual(set(f.get_ancestry(b'rM')), set(f.get_ancestry(b'rM')))

    def test_mutate_after_finish(self):
        self._transaction = 'before'
        f = self.get_file()
        self._transaction = 'after'
        self.assertRaises(errors.OutSideTransaction, f.add_lines, b'', [], [])
        self.assertRaises(errors.OutSideTransaction, f.add_lines_with_ghosts, b'', [], [])

    def test_copy_to(self):
        f = self.get_file()
        f.add_lines(b'0', [], [b'a\n'])
        t = MemoryTransport()
        f.copy_to('foo', t)
        for suffix in self.get_factory().get_suffixes():
            self.assertTrue(t.has('foo' + suffix))

    def test_get_suffixes(self):
        f = self.get_file()
        self.assertTrue(isinstance(self.get_factory().get_suffixes(), list))

    def test_get_parent_map(self):
        f = self.get_file()
        f.add_lines(b'r0', [], [b'a\n', b'b\n'])
        self.assertEqual({b'r0': ()}, f.get_parent_map([b'r0']))
        f.add_lines(b'r1', [b'r0'], [b'a\n', b'b\n'])
        self.assertEqual({b'r1': (b'r0',)}, f.get_parent_map([b'r1']))
        self.assertEqual({b'r0': (), b'r1': (b'r0',)}, f.get_parent_map([b'r0', b'r1']))
        f.add_lines(b'r2', [], [b'a\n', b'b\n'])
        f.add_lines(b'r3', [], [b'a\n', b'b\n'])
        f.add_lines(b'm', [b'r0', b'r1', b'r2', b'r3'], [b'a\n', b'b\n'])
        self.assertEqual({b'm': (b'r0', b'r1', b'r2', b'r3')}, f.get_parent_map([b'm']))
        self.assertEqual({}, f.get_parent_map(b'y'))
        self.assertEqual({b'r0': (), b'r1': (b'r0',)}, f.get_parent_map([b'r0', b'y', b'r1']))

    def test_annotate(self):
        f = self.get_file()
        f.add_lines(b'r0', [], [b'a\n', b'b\n'])
        f.add_lines(b'r1', [b'r0'], [b'c\n', b'b\n'])
        origins = f.annotate(b'r1')
        self.assertEqual(origins[0][0], b'r1')
        self.assertEqual(origins[1][0], b'r0')
        self.assertRaises(RevisionNotPresent, f.annotate, b'foo')

    def test_detection(self):
        w = self.get_file_corrupted_text()
        self.assertEqual(b'hello\n', w.get_text(b'v1'))
        self.assertRaises(WeaveInvalidChecksum, w.get_text, b'v2')
        self.assertRaises(WeaveInvalidChecksum, w.get_lines, b'v2')
        self.assertRaises(WeaveInvalidChecksum, w.check)
        w = self.get_file_corrupted_checksum()
        self.assertEqual(b'hello\n', w.get_text(b'v1'))
        self.assertRaises(WeaveInvalidChecksum, w.get_text, b'v2')
        self.assertRaises(WeaveInvalidChecksum, w.get_lines, b'v2')
        self.assertRaises(WeaveInvalidChecksum, w.check)

    def get_file_corrupted_text(self):
        """Return a versioned file with corrupt text but valid metadata."""
        raise NotImplementedError(self.get_file_corrupted_text)

    def reopen_file(self, name='foo'):
        """Open the versioned file from disk again."""
        raise NotImplementedError(self.reopen_file)

    def test_iter_lines_added_or_present_in_versions(self):

        class InstrumentedProgress(progress.ProgressTask):

            def __init__(self):
                progress.ProgressTask.__init__(self)
                self.updates = []

            def update(self, msg=None, current=None, total=None):
                self.updates.append((msg, current, total))
        vf = self.get_file()
        vf.add_lines(b'base', [], [b'base\n'])
        vf.add_lines(b'lancestor', [], [b'lancestor\n'])
        vf.add_lines(b'rancestor', [b'base'], [b'rancestor\n'])
        vf.add_lines(b'child', [b'rancestor'], [b'base\n', b'child\n'])
        vf.add_lines(b'otherchild', [b'lancestor', b'base'], [b'base\n', b'lancestor\n', b'otherchild\n'])

        def iter_with_versions(versions, expected):
            lines = {}
            progress = InstrumentedProgress()
            for line in vf.iter_lines_added_or_present_in_versions(versions, pb=progress):
                lines.setdefault(line, 0)
                lines[line] += 1
            if [] != progress.updates:
                self.assertEqual(expected, progress.updates)
            return lines
        lines = iter_with_versions([b'child', b'otherchild'], [('Walking content', 0, 2), ('Walking content', 1, 2), ('Walking content', 2, 2)])
        self.assertTrue(lines[b'child\n', b'child'] > 0)
        self.assertTrue(lines[b'otherchild\n', b'otherchild'] > 0)
        lines = iter_with_versions(None, [('Walking content', 0, 5), ('Walking content', 1, 5), ('Walking content', 2, 5), ('Walking content', 3, 5), ('Walking content', 4, 5), ('Walking content', 5, 5)])
        self.assertTrue(lines[b'base\n', b'base'] > 0)
        self.assertTrue(lines[b'lancestor\n', b'lancestor'] > 0)
        self.assertTrue(lines[b'rancestor\n', b'rancestor'] > 0)
        self.assertTrue(lines[b'child\n', b'child'] > 0)
        self.assertTrue(lines[b'otherchild\n', b'otherchild'] > 0)

    def test_add_lines_with_ghosts(self):
        vf = self.get_file()
        parent_id_unicode = 'b¿se'
        parent_id_utf8 = parent_id_unicode.encode('utf8')
        try:
            vf.add_lines_with_ghosts(b'notbxbfse', [parent_id_utf8], [])
        except NotImplementedError:
            self.assertRaises(NotImplementedError, vf.get_ancestry_with_ghosts, [b'foo'])
            self.assertRaises(NotImplementedError, vf.get_parents_with_ghosts, b'foo')
            return
        vf = self.reopen_file()
        self.assertEqual({b'notbxbfse'}, vf.get_ancestry(b'notbxbfse'))
        self.assertFalse(vf.has_version(parent_id_utf8))
        self.assertEqual({parent_id_utf8, b'notbxbfse'}, vf.get_ancestry_with_ghosts([b'notbxbfse']))
        self.assertEqual([parent_id_utf8], vf.get_parents_with_ghosts(b'notbxbfse'))
        vf.add_lines(parent_id_utf8, [], [])
        self.assertEqual({parent_id_utf8, b'notbxbfse'}, vf.get_ancestry([b'notbxbfse']))
        self.assertEqual({b'notbxbfse': (parent_id_utf8,)}, vf.get_parent_map([b'notbxbfse']))
        self.assertTrue(vf.has_version(parent_id_utf8))
        self.assertEqual({parent_id_utf8, b'notbxbfse'}, vf.get_ancestry_with_ghosts([b'notbxbfse']))
        self.assertEqual([parent_id_utf8], vf.get_parents_with_ghosts(b'notbxbfse'))

    def test_add_lines_with_ghosts_after_normal_revs(self):
        vf = self.get_file()
        try:
            vf.add_lines_with_ghosts(b'base', [], [b'line\n', b'line_b\n'])
        except NotImplementedError:
            return
        vf.add_lines_with_ghosts(b'references_ghost', [b'base', b'a_ghost'], [b'line\n', b'line_b\n', b'line_c\n'])
        origins = vf.annotate(b'references_ghost')
        self.assertEqual((b'base', b'line\n'), origins[0])
        self.assertEqual((b'base', b'line_b\n'), origins[1])
        self.assertEqual((b'references_ghost', b'line_c\n'), origins[2])

    def test_readonly_mode(self):
        t = self.get_transport()
        factory = self.get_factory()
        vf = factory('id', t, 511, create=True, access_mode='w')
        vf = factory('id', t, access_mode='r')
        self.assertRaises(errors.ReadOnlyError, vf.add_lines, b'base', [], [])
        self.assertRaises(errors.ReadOnlyError, vf.add_lines_with_ghosts, b'base', [], [])

    def test_get_sha1s(self):
        vf = self.get_file()
        vf.add_lines(b'a', [], [b'a\n'])
        vf.add_lines(b'b', [b'a'], [b'a\n'])
        vf.add_lines(b'c', [], [b'a'])
        self.assertEqual({b'a': b'3f786850e387550fdab836ed7e6dc881de23001b', b'c': b'86f7e437faa5a7fce15d1ddcb9eaeaea377667b8', b'b': b'3f786850e387550fdab836ed7e6dc881de23001b'}, vf.get_sha1s([b'a', b'c', b'b']))