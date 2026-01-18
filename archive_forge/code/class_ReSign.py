from breezy import gpg, tests
from breezy.bzr.testament import Testament
from breezy.controldir import ControlDir
class ReSign(tests.TestCaseInTempDir):

    def monkey_patch_gpg(self):
        """Monkey patch the gpg signing strategy to be a loopback.

        This also registers the cleanup, so that we will revert to
        the original gpg strategy when done.
        """
        self.overrideAttr(gpg, 'GPGStrategy', gpg.LoopbackGPGStrategy)

    def setup_tree(self):
        wt = ControlDir.create_standalone_workingtree('.')
        a = wt.commit('base A', allow_pointless=True)
        b = wt.commit('base B', allow_pointless=True)
        c = wt.commit('base C', allow_pointless=True)
        return (wt, [a, b, c])

    def assertEqualSignature(self, repo, revision_id):
        """Assert a signature is stored correctly in repository."""
        self.assertEqual(b'-----BEGIN PSEUDO-SIGNED CONTENT-----\n' + Testament.from_revision(repo, revision_id).as_short_text() + b'-----END PSEUDO-SIGNED CONTENT-----\n', repo.get_signature_text(revision_id))

    def test_resign(self):
        wt, [a, b, c] = self.setup_tree()
        repo = wt.branch.repository
        self.monkey_patch_gpg()
        self.run_bzr('re-sign -r revid:%s' % a.decode('utf-8'))
        self.assertEqualSignature(repo, a)
        self.run_bzr('re-sign %s' % b.decode('utf-8'))
        self.assertEqualSignature(repo, b)

    def test_resign_range(self):
        wt, [a, b, c] = self.setup_tree()
        repo = wt.branch.repository
        self.monkey_patch_gpg()
        self.run_bzr('re-sign -r 1..')
        self.assertEqualSignature(repo, a)
        self.assertEqualSignature(repo, b)
        self.assertEqualSignature(repo, c)

    def test_resign_multiple(self):
        wt, rs = self.setup_tree()
        repo = wt.branch.repository
        self.monkey_patch_gpg()
        self.run_bzr('re-sign ' + ' '.join((r.decode('utf-8') for r in rs)))
        for r in rs:
            self.assertEqualSignature(repo, r)

    def test_resign_directory(self):
        """Test --directory option"""
        wt = ControlDir.create_standalone_workingtree('a')
        a = wt.commit('base A', allow_pointless=True)
        b = wt.commit('base B', allow_pointless=True)
        c = wt.commit('base C', allow_pointless=True)
        repo = wt.branch.repository
        self.monkey_patch_gpg()
        self.run_bzr('re-sign --directory=a -r revid:' + a.decode('utf-8'))
        self.assertEqualSignature(repo, a)
        self.run_bzr('re-sign -d a %s' % b.decode('utf-8'))
        self.assertEqualSignature(repo, b)