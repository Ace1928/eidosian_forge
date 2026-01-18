import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestShowLog(tests.TestCaseWithTransport):

    def checkDelta(self, delta, **kw):
        """Check the filenames touched by a delta are as expected.

        Caller only have to pass in the list of files for each part, all
        unspecified parts are considered empty (and checked as such).
        """
        for n in ('added', 'removed', 'renamed', 'modified', 'unchanged'):
            expected = kw.get(n, [])
            got = [x.path[1] or x.path[0] for x in getattr(delta, n)]
            self.assertEqual(expected, got)

    def assertInvalidRevisonNumber(self, br, start, end):
        lf = LogCatcher()
        self.assertRaises(errors.InvalidRevisionNumber, log.show_log, br, lf, start_revision=start, end_revision=end)

    def test_cur_revno(self):
        wt = self.make_branch_and_tree('.')
        b = wt.branch
        lf = LogCatcher()
        wt.commit('empty commit')
        log.show_log(b, lf, verbose=True, start_revision=1, end_revision=1)
        self.assertInvalidRevisonNumber(b, 2, 1)
        self.assertInvalidRevisonNumber(b, 1, 2)
        self.assertInvalidRevisonNumber(b, 0, 2)
        self.assertInvalidRevisonNumber(b, -1, 1)
        self.assertInvalidRevisonNumber(b, 1, -1)
        self.assertInvalidRevisonNumber(b, 1, 0)

    def test_empty_branch(self):
        wt = self.make_branch_and_tree('.')
        lf = LogCatcher()
        log.show_log(wt.branch, lf)
        self.assertEqual([], lf.revisions)

    def test_empty_commit(self):
        wt = self.make_branch_and_tree('.')
        wt.commit('empty commit')
        lf = LogCatcher()
        log.show_log(wt.branch, lf, verbose=True)
        revs = lf.revisions
        self.assertEqual(1, len(revs))
        self.assertEqual('1', revs[0].revno)
        self.assertEqual('empty commit', revs[0].rev.message)
        self.checkDelta(revs[0].delta)

    def test_simple_commit(self):
        wt = self.make_branch_and_tree('.')
        wt.commit('empty commit')
        self.build_tree(['hello'])
        wt.add('hello')
        wt.commit('add one file', committer='Ľórém Ípšúm <test@example.com>')
        lf = LogCatcher()
        log.show_log(wt.branch, lf, verbose=True)
        self.assertEqual(2, len(lf.revisions))
        log_entry = lf.revisions[0]
        self.assertEqual('2', log_entry.revno)
        self.assertEqual('add one file', log_entry.rev.message)
        self.checkDelta(log_entry.delta, added=['hello'])

    def test_commit_message_with_control_chars(self):
        wt = self.make_branch_and_tree('.')
        msg = 'All 8-bit chars: ' + ''.join([chr(x) for x in range(256)])
        msg = msg.replace('\r', '\n')
        wt.commit(msg)
        lf = LogCatcher()
        log.show_log(wt.branch, lf, verbose=True)
        committed_msg = lf.revisions[0].rev.message
        if wt.branch.repository._serializer.squashes_xml_invalid_characters:
            self.assertNotEqual(msg, committed_msg)
            self.assertTrue(len(committed_msg) > len(msg))
        else:
            self.assertEqual(msg, committed_msg)

    def test_commit_message_without_control_chars(self):
        wt = self.make_branch_and_tree('.')
        msg = '\t' + ''.join([chr(x) for x in range(32, 256)])
        wt.commit(msg)
        lf = LogCatcher()
        log.show_log(wt.branch, lf, verbose=True)
        committed_msg = lf.revisions[0].rev.message
        self.assertEqual(msg, committed_msg)

    def test_deltas_in_merge_revisions(self):
        """Check deltas created for both mainline and merge revisions"""
        wt = self.make_branch_and_tree('parent')
        self.build_tree(['parent/file1', 'parent/file2', 'parent/file3'])
        wt.add('file1')
        wt.add('file2')
        wt.commit(message='add file1 and file2')
        self.run_bzr('branch parent child')
        os.unlink('child/file1')
        with open('child/file2', 'wb') as f:
            f.write(b'hello\n')
        self.run_bzr(['commit', '-m', 'remove file1 and modify file2', 'child'])
        os.chdir('parent')
        self.run_bzr('merge ../child')
        wt.commit('merge child branch')
        os.chdir('..')
        b = wt.branch
        lf = LogCatcher()
        lf.supports_merge_revisions = True
        log.show_log(b, lf, verbose=True)
        revs = lf.revisions
        self.assertEqual(3, len(revs))
        logentry = revs[0]
        self.assertEqual('2', logentry.revno)
        self.assertEqual('merge child branch', logentry.rev.message)
        self.checkDelta(logentry.delta, removed=['file1'], modified=['file2'])
        logentry = revs[1]
        self.assertEqual('1.1.1', logentry.revno)
        self.assertEqual('remove file1 and modify file2', logentry.rev.message)
        self.checkDelta(logentry.delta, removed=['file1'], modified=['file2'])
        logentry = revs[2]
        self.assertEqual('1', logentry.revno)
        self.assertEqual('add file1 and file2', logentry.rev.message)
        self.checkDelta(logentry.delta, added=['file1', 'file2'])

    def test_bug_842695_log_restricted_to_dir(self):
        trunk = self.make_branch_and_tree('this')
        trunk.commit('initial trunk')
        adder = trunk.controldir.sprout('adder').open_workingtree()
        merger = trunk.controldir.sprout('merger').open_workingtree()
        self.build_tree_contents([('adder/dir/',), ('adder/dir/file', b'foo')])
        adder.add(['dir', 'dir/file'])
        adder.commit('added dir')
        trunk.merge_from_branch(adder.branch)
        trunk.commit('merged adder into trunk')
        merger.merge_from_branch(trunk.branch)
        merger.commit('merged trunk into merger')
        for i in range(200):
            trunk.commit('intermediate commit %d' % i)
        trunk.merge_from_branch(merger.branch)
        trunk.commit('merged merger into trunk')
        file_id = trunk.path2id('dir')
        lf = LogCatcher()
        lf.supports_merge_revisions = True
        log.show_log(trunk.branch, lf, file_id)
        try:
            self.assertEqual(['2', '1.1.1'], [r.revno for r in lf.revisions])
        except AssertionError:
            raise tests.KnownFailure('bug #842695')