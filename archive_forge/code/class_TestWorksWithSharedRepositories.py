from breezy import errors, tests
from breezy.bzr import bzrdir
from breezy.reconcile import Reconciler, reconcile
from breezy.tests import per_repository
class TestWorksWithSharedRepositories(per_repository.TestCaseWithRepository):

    def test_reweave_empty(self):
        parent = bzrdir.BzrDirMetaFormat1().initialize('.')
        parent.create_repository(shared=True)
        parent.root_transport.mkdir('child')
        child = bzrdir.BzrDirMetaFormat1().initialize('child')
        self.assertRaises(errors.NoRepositoryPresent, child.open_repository)
        reconciler = Reconciler(child)
        result = reconciler.reconcile()
        reconcile(child)
        self.assertEqual(0, result.inconsistent_parents)
        self.assertEqual(0, result.garbage_inventories)