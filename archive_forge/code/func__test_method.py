from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
def _test_method(self, name, base_result, *args, **kwargs):
    self.fake_repo.result = base_result
    method = getattr(self.proxy_repo, name)
    proxy_result = method(*args, **kwargs)
    self.assertIsInstance(proxy_result, FakeProxy)
    self.assertEqual(base_result, proxy_result.base)
    self.assertEqual(0, len(proxy_result.args))
    self.assertEqual({'a': 1}, proxy_result.kwargs)
    self.assertEqual(args, self.fake_repo.args)
    self.assertEqual(kwargs, self.fake_repo.kwargs)