from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
def _test_method_with_proxied_argument(self, name, result, **kwargs):
    self.fake_repo.result = result
    item = FakeProxy('snoop')
    method = getattr(self.proxy_repo, name)
    proxy_result = method(item)
    self.assertEqual(('snoop',), self.fake_repo.args)
    self.assertEqual(kwargs, self.fake_repo.kwargs)
    if result is None:
        self.assertIsNone(proxy_result)
    else:
        self.assertIsInstance(proxy_result, FakeProxy)
        self.assertEqual(result, proxy_result.base)
        self.assertEqual(tuple(), proxy_result.args)
        self.assertEqual({'a': 1}, proxy_result.kwargs)