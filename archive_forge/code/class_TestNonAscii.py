import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestNonAscii(tests.TestCaseWithTransport):
    """Test that brz handles files/committers/etc which are non-ascii."""
    scenarios = EncodingAdapter.encoding_scenarios

    def setUp(self):
        super().setUp()
        self._check_can_encode_paths()
        self.overrideAttr(osutils, '_cached_user_encoding', self.encoding)
        email = self.info['committer'] + ' <joe@foo.com>'
        self.overrideEnv('BRZ_EMAIL', email)
        self.create_base()

    def run_bzr_decode(self, args, encoding=None, fail=False, retcode=None, working_dir=None):
        """Run brz and decode the output into a particular encoding.

        Returns a string containing the stdout output from bzr.

        :param fail: If true, the operation is expected to fail with
            a UnicodeError.
        """
        if encoding is None:
            encoding = osutils.get_user_encoding()
        try:
            out = self.run_bzr_raw(args, encoding=encoding, retcode=retcode, working_dir=working_dir)[0]
            return out.decode(encoding)
        except UnicodeError as e:
            if not fail:
                raise
        else:
            if fail:
                self.fail('Expected UnicodeError not raised')

    def _check_OSX_can_roundtrip(self, path, fs_enc=None):
        """Stop the test if it's about to fail or errors out.

        Until we get proper support on OSX for accented paths (in fact, any
        path whose NFD decomposition is different than the NFC one), this is
        the best way to keep test active (as opposed to disabling them
        completely). This is a stop gap. The tests should at least be rewritten
        so that the failing ones are clearly separated from the passing ones.
        """
        if fs_enc is None:
            fs_enc = sys.getfilesystemencoding()
        if sys.platform == 'darwin':
            encoded = path.encode(fs_enc)
            import unicodedata
            normal_thing = unicodedata.normalize('NFD', path)
            mac_encoded = normal_thing.encode(fs_enc)
            if mac_encoded != encoded:
                self.knownFailure('Unable to roundtrip path %r on OSX filesystem using encoding "%s"' % (path, fs_enc))

    def _check_can_encode_paths(self):
        fs_enc = sys.getfilesystemencoding()
        terminal_enc = osutils.get_terminal_encoding()
        fname = self.info['filename']
        dir_name = self.info['directory']
        for thing in [fname, dir_name]:
            try:
                thing.encode(fs_enc)
            except UnicodeEncodeError:
                raise tests.TestSkipped('Unable to represent path %r in filesystem encoding "%s"' % (thing, fs_enc))
            try:
                thing.encode(terminal_enc)
            except UnicodeEncodeError:
                raise tests.TestSkipped('Unable to represent path %r in terminal encoding "%s" (even though it is valid in filesystem encoding "%s")' % (thing, terminal_enc, fs_enc))

    def create_base(self):
        wt = self.make_branch_and_tree('.')
        self.build_tree_contents([('a', b'foo\n')])
        wt.add('a')
        wt.commit('adding a')
        self.build_tree_contents([('b', b'non-ascii \xff\xff\xfc\xfb\x00 in b\n')])
        wt.add('b')
        wt.commit(self.info['message'])
        fname = self.info['filename']
        self.build_tree_contents([(fname, b'unicode filename\n')])
        wt.add(fname)
        wt.commit('And a unicode file\n')
        self.wt = wt

    def test_status(self):
        self.build_tree_contents([(self.info['filename'], b'changed something\n')])
        txt = self.run_bzr_decode('status')
        self._check_OSX_can_roundtrip(self.info['filename'])
        self.assertEqual('modified:\n  {}\n'.format(self.info['filename']), txt)
        txt = self.run_bzr_decode('status', encoding='ascii')
        expected = 'modified:\n  {}\n'.format(self.info['filename'].encode('ascii', 'replace').decode('ascii'))
        self.assertEqual(expected, txt)

    def test_cat(self):
        txt = self.run_bzr_raw('cat b')[0]
        self.assertEqual(b'non-ascii \xff\xff\xfc\xfb\x00 in b\n', txt)
        self._check_OSX_can_roundtrip(self.info['filename'])
        txt = self.run_bzr_raw(['cat', self.info['filename']])[0]
        self.assertEqual(b'unicode filename\n', txt)

    def test_cat_revision(self):
        committer = self.info['committer']
        txt = self.run_bzr_decode('cat-revision -r 1')
        self.assertTrue(committer in txt, 'failed to find {!r} in {!r}'.format(committer, txt))
        msg = self.info['message']
        txt = self.run_bzr_decode('cat-revision -r 2')
        self.assertTrue(msg in txt, 'failed to find {!r} in {!r}'.format(msg, txt))

    def test_mkdir(self):
        txt = self.run_bzr_decode(['mkdir', self.info['directory']])
        self.assertEqual('added %s\n' % self.info['directory'], txt)
        txt = self.run_bzr_raw(['mkdir', self.info['directory'] + '2'], encoding='ascii')[0]
        expected = 'added {}2\n'.format(self.info['directory'])
        expected = expected.encode('ascii', 'replace')
        self.assertEqual(expected, txt)

    def test_relpath(self):
        txt = self.run_bzr_decode(['relpath', self.info['filename']])
        self.assertEqual(self.info['filename'] + '\n', txt)
        self.run_bzr_decode(['relpath', self.info['filename']], encoding='ascii', fail=True)

    def test_inventory(self):
        txt = self.run_bzr_decode('inventory')
        self._check_OSX_can_roundtrip(self.info['filename'])
        self.assertEqual(['a', 'b', self.info['filename']], txt.splitlines())
        self.run_bzr_decode('inventory', encoding='ascii', fail=True)
        txt = self.run_bzr_decode('inventory --show-ids')

    def test_revno(self):
        self.assertEqual('3\n', self.run_bzr_decode('revno'))
        self.assertEqual('3\n', self.run_bzr_decode('revno', encoding='ascii'))

    def test_revision_info(self):
        self.run_bzr_decode('revision-info -r 1')
        self.run_bzr_decode('revision-info -r 1', encoding='ascii')

    def test_mv(self):
        fname1 = self.info['filename']
        fname2 = self.info['filename'] + '2'
        dirname = self.info['directory']
        self.run_bzr_decode(['mv', 'a', fname1], fail=True)
        txt = self.run_bzr_decode(['mv', 'a', fname2])
        self.assertEqual('a => %s\n' % fname2, txt)
        self.assertPathDoesNotExist('a')
        self.assertPathExists(fname2)
        self.wt = self.wt.controldir.open_workingtree()
        self.wt.commit('renamed to non-ascii')
        os.mkdir(dirname)
        self.wt.add(dirname)
        txt = self.run_bzr_decode(['mv', fname1, fname2, dirname])
        self._check_OSX_can_roundtrip(self.info['filename'])
        self.assertEqual(['{} => {}/{}'.format(fname1, dirname, fname1), '{} => {}/{}'.format(fname2, dirname, fname2)], txt.splitlines())
        newpath = '{}/{}'.format(dirname, fname2)
        txt = self.run_bzr_raw(['mv', newpath, 'a'], encoding='ascii')[0]
        self.assertPathExists('a')
        self.assertEqual(newpath.encode('ascii', 'replace') + b' => a\n', txt)

    def test_branch(self):
        self.run_bzr_decode(['branch', '.', self.info['directory']])
        self.run_bzr_decode(['branch', '.', self.info['directory'] + '2'], encoding='ascii')

    def test_pull(self):
        dirname1 = self.info['directory']
        dirname2 = self.info['directory'] + '2'
        url1 = urlutils.local_path_to_url(dirname1)
        url2 = urlutils.local_path_to_url(dirname2)
        out_bzrdir = self.wt.controldir.sprout(url1)
        out_bzrdir.sprout(url2)
        self.build_tree_contents([(osutils.pathjoin(dirname1, 'a'), b'different text\n')])
        self.wt.commit('mod a')
        txt = self.run_bzr_decode('pull', working_dir=dirname2)
        expected = osutils.pathjoin(osutils.getcwd(), dirname1)
        self.assertEqual('Using saved parent location: %s/\nNo revisions or tags to pull.\n' % (expected,), txt)
        self.build_tree_contents([(osutils.pathjoin(dirname1, 'a'), b'and yet more\n')])
        self.wt.commit('modifying a by ' + self.info['committer'])
        self.run_bzr_decode('pull --verbose', encoding='ascii', working_dir=dirname2)

    def test_push(self):
        dirname = self.info['directory']
        self.run_bzr_decode(['push', dirname])
        self.build_tree_contents([('a', b'adding more text\n')])
        self.wt.commit('added some stuff')
        self.run_bzr_decode('push')
        self.build_tree_contents([('a', b'and a bit more: \n%s\n' % (dirname.encode('utf-8'),))])
        self.wt.commit('Added some ' + dirname)
        self.run_bzr_decode('push --verbose', encoding='ascii')
        self.run_bzr_decode(['push', '--verbose', dirname + '2'])
        self.run_bzr_decode(['push', '--verbose', dirname + '3'], encoding='ascii')
        self.run_bzr_decode(['push', '--verbose', '--create-prefix', dirname + '4/' + dirname + '5'])
        self.run_bzr_decode(['push', '--verbose', '--create-prefix', dirname + '6/' + dirname + '7'], encoding='ascii')

    def test_renames(self):
        fname = self.info['filename'] + '2'
        self.wt.rename_one('a', fname)
        txt = self.run_bzr_decode('renames')
        self.assertEqual('a => %s\n' % fname, txt)
        self.run_bzr_decode('renames', fail=True, encoding='ascii')

    def test_remove(self):
        fname = self.info['filename']
        txt = self.run_bzr_decode(['remove', fname], encoding='ascii')

    def test_remove_verbose(self):
        fname = self.info['filename']
        txt = self.run_bzr_decode(['remove', '--verbose', fname], encoding='ascii')

    def test_file_id(self):
        fname = self.info['filename']
        txt = self.run_bzr_decode(['file-id', fname])
        txt = self.run_bzr_decode(['file-id', fname], encoding='ascii')

    def test_file_path(self):
        fname = self.info['filename']
        dirname = self.info['directory']
        self.build_tree_contents([('base/',), (osutils.pathjoin('base', '{}/'.format(dirname)),)])
        self.wt.add('base')
        self.wt.add('base/' + dirname)
        path = osutils.pathjoin('base', dirname, fname)
        self._check_OSX_can_roundtrip(self.info['filename'])
        self.wt.rename_one(fname, path)
        self.wt.commit('moving things around')
        txt = self.run_bzr_decode(['file-path', path])
        txt = self.run_bzr_decode(['file-path', path], encoding='ascii')

    def test_revision_history(self):
        txt = self.run_bzr_decode('revision-history')

    def test_ancestry(self):
        txt = self.run_bzr_decode('ancestry')

    def test_diff(self):
        self._check_OSX_can_roundtrip(self.info['filename'])
        self.build_tree_contents([(self.info['filename'], b'newline\n')])
        txt = self.run_bzr('diff', retcode=1)[0]

    def test_deleted(self):
        self._check_OSX_can_roundtrip(self.info['filename'])
        fname = self.info['filename']
        os.remove(fname)
        self.wt.remove(fname)
        txt = self.run_bzr_decode('deleted')
        self.assertEqual(fname + '\n', txt)
        txt = self.run_bzr_decode('deleted --show-ids')
        self.assertTrue(txt.startswith(fname))
        self.run_bzr_decode('deleted', encoding='ascii', fail=True)

    def test_modified(self):
        fname = self.info['filename']
        self.build_tree_contents([(fname, b'modified\n')])
        txt = self.run_bzr_decode('modified')
        self._check_OSX_can_roundtrip(self.info['filename'])
        self.assertEqual('"' + fname + '"' + '\n', txt)
        self.run_bzr_decode('modified', encoding='ascii', fail=True)

    def test_added(self):
        fname = self.info['filename'] + '2'
        self.build_tree_contents([(fname, b'added\n')])
        self.wt.add(fname)
        txt = self.run_bzr_decode('added')
        self.assertEqual('"' + fname + '"' + '\n', txt)
        self.run_bzr_decode('added', encoding='ascii', fail=True)

    def test_root(self):
        dirname = self.info['directory']
        url = urlutils.local_path_to_url(dirname)
        self.run_bzr_decode('root')
        self.wt.controldir.sprout(url)
        txt = self.run_bzr_decode('root', working_dir=dirname)
        self.assertTrue(txt.endswith(dirname + '\n'))
        txt = self.run_bzr_decode('root', encoding='ascii', fail=True, working_dir=dirname)

    def test_log(self):
        fname = self.info['filename']
        txt = self.run_bzr_decode('log')
        self.assertNotEqual(-1, txt.find(self.info['committer']))
        self.assertNotEqual(-1, txt.find(self.info['message']))
        txt = self.run_bzr_decode('log --verbose')
        self._check_OSX_can_roundtrip(self.info['filename'])
        self.assertNotEqual(-1, txt.find(fname))
        txt = self.run_bzr_raw('log --verbose', encoding='ascii')[0]
        self.assertNotEqual(-1, txt.find(fname.encode('ascii', 'replace')))

    def test_touching_revisions(self):
        fname = self.info['filename']
        txt = self.run_bzr_decode(['touching-revisions', fname])
        self._check_OSX_can_roundtrip(self.info['filename'])
        self.assertEqual('     3 added {}\n'.format(fname), txt)
        fname2 = self.info['filename'] + '2'
        self.wt.rename_one(fname, fname2)
        self.wt.commit('Renamed {} => {}'.format(fname, fname2))
        txt = self.run_bzr_decode(['touching-revisions', fname2])
        expected_txt = '     3 added %s\n     4 renamed %s => %s\n' % (fname, fname, fname2)
        self.assertEqual(expected_txt, txt)
        self.run_bzr_decode(['touching-revisions', fname2], encoding='ascii', fail=True)

    def test_ls(self):
        txt = self.run_bzr_decode('ls')
        self.assertEqual(sorted(['a', 'b', self.info['filename']]), sorted(txt.splitlines()))
        txt = self.run_bzr_decode('ls --null')
        self.assertEqual(sorted(['', 'a', 'b', self.info['filename']]), sorted(txt.split('\x00')))
        txt = self.run_bzr_decode('ls', encoding='ascii', fail=True)
        txt = self.run_bzr_decode('ls --null', encoding='ascii', fail=True)

    def test_unknowns(self):
        fname = self.info['filename'] + '2'
        self.build_tree_contents([(fname, b'unknown\n')])
        txt = self.run_bzr_decode('unknowns')
        self._check_OSX_can_roundtrip(self.info['filename'])
        self.assertEqual('"{}"\n'.format(fname), txt)
        self.run_bzr_decode('unknowns', encoding='ascii', fail=True)

    def test_ignore(self):
        fname2 = self.info['filename'] + '2.txt'
        self.build_tree_contents([(fname2, b'ignored\n')])

        def check_unknowns(expected):
            self.assertEqual(expected, list(self.wt.unknowns()))
        self._check_OSX_can_roundtrip(self.info['filename'])
        check_unknowns([fname2])
        self.run_bzr_decode(['ignore', './' + fname2])
        check_unknowns([])
        fname3 = self.info['filename'] + '3.txt'
        self.build_tree_contents([(fname3, b'unknown 3\n')])
        check_unknowns([fname3])
        self.run_bzr_decode(['ignore', fname3], encoding='ascii')
        check_unknowns([])
        fname4 = self.info['filename'] + '4.txt'
        self.build_tree_contents([(fname4, b'unknown 4\n')])
        self.run_bzr_decode('ignore *.txt')
        check_unknowns([])
        os.remove('.bzrignore')
        self.run_bzr_decode(['ignore', self.info['filename'] + '*'])
        check_unknowns([])

    def test_missing(self):
        self.make_branch_and_tree('empty-tree')
        msg = self.info['message']
        txt = self.run_bzr_decode('missing empty-tree')
        self.assertNotEqual(-1, txt.find(self.info['committer']))
        self.assertNotEqual(-1, txt.find(msg))
        txt = self.run_bzr_raw('missing empty-tree', encoding='ascii', retcode=1)[0]
        self.assertNotEqual(-1, txt.find(msg.encode('ascii', 'replace')))

    def test_info(self):
        self.run_bzr_decode(['branch', '.', self.info['directory']])
        self.run_bzr_decode(['info', self.info['directory']])
        self.run_bzr_decode(['info', self.info['directory']], encoding='ascii')

    def test_ignored(self):
        fname = self.info['filename'] + '1.txt'
        self.build_tree_contents([(fname, b'ignored\n')])
        self.run_bzr(['ignore', fname])
        txt = self.run_bzr_decode(['ignored'])
        self.assertEqual(txt, '%-50s %s\n' % (fname, fname))
        txt = self.run_bzr_decode(['ignored'], encoding='ascii')
        fname = fname.encode('ascii', 'replace').decode('ascii')
        self.assertEqual(txt, '%-50s %s\n' % (fname, fname))