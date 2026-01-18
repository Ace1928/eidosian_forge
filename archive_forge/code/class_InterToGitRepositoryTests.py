from ...controldir import format_registry
from ...repository import InterRepository
from ...tests import TestCaseWithTransport
from ..interrepo import InterToGitRepository
from ..mapping import BzrGitMappingExperimental, BzrGitMappingv1
class InterToGitRepositoryTests(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.git_repo = self.make_repository('git', format=format_registry.make_controldir('git'))
        self.bzr_repo = self.make_repository('bzr', shared=True)

    def _get_interrepo(self, mapping=None):
        self.bzr_repo.lock_read()
        self.addCleanup(self.bzr_repo.unlock)
        interrepo = InterRepository.get(self.bzr_repo, self.git_repo)
        if mapping is not None:
            interrepo.mapping = mapping
        return interrepo

    def test_instance(self):
        self.assertIsInstance(self._get_interrepo(), InterToGitRepository)

    def test_pointless_fetch_refs_old_mapping(self):
        interrepo = self._get_interrepo(mapping=BzrGitMappingv1())
        interrepo.fetch_refs(lambda x: {}, lossy=False)

    def test_pointless_fetch_refs(self):
        interrepo = self._get_interrepo(mapping=BzrGitMappingExperimental())
        revidmap, old_refs, new_refs = interrepo.fetch_refs(lambda x: {}, lossy=False)
        self.assertEqual(old_refs, {b'HEAD': (b'ref: refs/heads/master', None)})
        self.assertEqual(new_refs, {})

    def test_pointless_lossy_fetch_refs(self):
        revidmap, old_refs, new_refs = self._get_interrepo().fetch_refs(lambda x: {}, lossy=True)
        self.assertEqual(old_refs, {b'HEAD': (b'ref: refs/heads/master', None)})
        self.assertEqual(new_refs, {})
        self.assertEqual(revidmap, {})

    def test_pointless_missing_revisions(self):
        interrepo = self._get_interrepo()
        interrepo.source_store.lock_read()
        self.addCleanup(interrepo.source_store.unlock)
        self.assertEqual([], list(interrepo.missing_revisions([])))

    def test_missing_revisions_unknown_stop_rev(self):
        interrepo = self._get_interrepo()
        interrepo.source_store.lock_read()
        self.addCleanup(interrepo.source_store.unlock)
        self.assertEqual([], list(interrepo.missing_revisions([(None, b'unknown')])))

    def test_odd_rename(self):
        branch = self.bzr_repo.controldir.create_branch()
        tree = branch.controldir.create_workingtree()
        self.build_tree(['bzr/bar/', 'bzr/bar/foobar'])
        tree.add(['bar', 'bar/foobar'])
        tree.commit('initial')
        self.build_tree(['bzr/baz/'])
        tree.add(['baz'])
        tree.rename_one('bar', 'baz/IrcDotNet')
        last_revid = tree.commit('rename')

        def decide(x):
            return {b'refs/heads/master': (None, last_revid)}
        interrepo = self._get_interrepo()
        revidmap, old_refs, new_refs = interrepo.fetch_refs(decide, lossy=True)
        gitid = revidmap[last_revid][0]
        store = self.git_repo._git.object_store
        commit = store[gitid]
        tree = store[commit.tree]
        tree.check()
        self.assertIn(b'baz', tree, repr(tree.items()))
        self.assertIn(tree[b'baz'][1], store)
        baz = store[tree[b'baz'][1]]
        baz.check()
        ircdotnet = store[baz[b'IrcDotNet'][1]]
        ircdotnet.check()
        foobar = store[ircdotnet[b'foobar'][1]]
        foobar.check()