from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
class TestProxyRepoPlain(test_utils.BaseTestCase):

    def setUp(self):
        super(TestProxyRepoPlain, self).setUp()
        self.fake_repo = FakeRepo()
        self.proxy_repo = proxy.Repo(self.fake_repo)

    def _test_method(self, name, base_result, *args, **kwargs):
        self.fake_repo.result = base_result
        method = getattr(self.proxy_repo, name)
        proxy_result = method(*args, **kwargs)
        self.assertEqual(base_result, proxy_result)
        self.assertEqual(args, self.fake_repo.args)
        self.assertEqual(kwargs, self.fake_repo.kwargs)

    def test_get(self):
        self._test_method('get', 'snarf', 'abcd')

    def test_list(self):
        self._test_method('list', ['sniff', 'snarf'], 2, filter='^sn')

    def test_add(self):
        self._test_method('add', 'snuff', 'enough')

    def test_save(self):
        self._test_method('save', 'snuff', 'enough', from_state=None)

    def test_remove(self):
        self._test_method('add', None, 'flying')

    def test_set_property_atomic(self):
        image = mock.MagicMock()
        image.image_id = 'foo'
        self._test_method('set_property_atomic', None, image, 'foo', 'bar')

    def test_set_property_nonimage(self):
        self.assertRaises(AssertionError, self._test_method, 'set_property_atomic', None, 'notimage', 'foo', 'bar')

    def test_delete_property_atomic(self):
        image = mock.MagicMock()
        image.image_id = 'foo'
        self._test_method('delete_property_atomic', None, image, 'foo', 'bar')

    def test_delete_property_nonimage(self):
        self.assertRaises(AssertionError, self._test_method, 'delete_property_atomic', None, 'notimage', 'foo', 'bar')