from unittest import mock
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import scheduler_stats
class PoolManagerTest(utils.TestCase):

    def setUp(self):
        super(PoolManagerTest, self).setUp()
        self.manager = scheduler_stats.PoolManager(fakes.FakeClient())

    @mock.patch.object(scheduler_stats.PoolManager, '_list', mock.Mock())
    def test_list(self):
        self.manager.list(detailed=False)
        self.manager._list.assert_called_once_with(scheduler_stats.RESOURCES_PATH, scheduler_stats.RESOURCES_NAME)

    @mock.patch.object(scheduler_stats.PoolManager, '_list', mock.Mock())
    def test_list_detail(self):
        self.manager.list()
        self.manager._list.assert_called_once_with(scheduler_stats.RESOURCES_PATH + '/detail', scheduler_stats.RESOURCES_NAME)

    @mock.patch.object(scheduler_stats.PoolManager, '_list', mock.Mock())
    def test_list_with_one_search_opt(self):
        host = 'fake_host'
        query_string = '?host=%s' % host
        self.manager.list(detailed=False, search_opts={'host': host})
        self.manager._list.assert_called_once_with(scheduler_stats.RESOURCES_PATH + query_string, scheduler_stats.RESOURCES_NAME)

    @mock.patch.object(scheduler_stats.PoolManager, '_list', mock.Mock())
    def test_list_detail_with_two_search_opts(self):
        host = 'fake_host'
        backend = 'fake_backend'
        query_string = '?backend=%s&host=%s' % (backend, host)
        self.manager.list(search_opts={'host': host, 'backend': backend})
        self.manager._list.assert_called_once_with(scheduler_stats.RESOURCES_PATH + '/detail' + query_string, scheduler_stats.RESOURCES_NAME)