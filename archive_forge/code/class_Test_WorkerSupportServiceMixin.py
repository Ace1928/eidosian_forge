import abc
from neutron_lib.services import base
from neutron_lib.tests import _base as test_base
class Test_WorkerSupportServiceMixin(test_base.BaseTestCase):

    def setUp(self):
        super(Test_WorkerSupportServiceMixin, self).setUp()
        self.worker = _Worker()

    def test_allocate_workers(self):
        self.assertEqual([], self.worker.get_workers())

    def test_add_worker(self):
        workers = [object(), object()]
        for w in workers:
            self.worker.add_worker(w)
        self.assertSequenceEqual(workers, self.worker.get_workers())

    def test_add_workers(self):
        workers = [object(), object(), object()]
        self.worker.add_workers(workers)
        self.assertSequenceEqual(workers, self.worker.get_workers())