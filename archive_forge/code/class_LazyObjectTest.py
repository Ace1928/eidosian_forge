import sys
import unittest
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.base import Response, LazyObject
from libcloud.common.exceptions import BaseHTTPError, RateLimitReachedError
class LazyObjectTest(LibcloudTestCase):

    class A(LazyObject):

        def __init__(self, x, y=None):
            self.x = x
            self.y = y

    def test_lazy_init(self):
        a = self.A(1, y=2)
        self.assertTrue(isinstance(a, self.A))
        with mock.patch.object(self.A, '__init__', return_value=None) as mock_init:
            a = self.A.lazy(3, y=4)
            self.assertTrue(isinstance(a, self.A))
            mock_init.assert_not_called()
            self.assertEqual(a.__dict__, {})
            mock_init.assert_called_once_with(3, y=4)

    def test_setattr(self):
        a = self.A.lazy('foo', y='bar')
        a.z = 'baz'
        wrapped_lazy_obj = object.__getattribute__(a, '_lazy_obj')
        self.assertEqual(a.z, 'baz')
        self.assertEqual(wrapped_lazy_obj.z, 'baz')