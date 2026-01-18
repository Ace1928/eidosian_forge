import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
class CachedPropertyTest(test.TestCase):

    def test_attribute_caching(self):

        class A(object):

            def __init__(self):
                self.call_counter = 0

            @misc.cachedproperty
            def b(self):
                self.call_counter += 1
                return 'b'
        a = A()
        self.assertEqual('b', a.b)
        self.assertEqual('b', a.b)
        self.assertEqual(1, a.call_counter)

    def test_custom_property(self):

        class A(object):

            @misc.cachedproperty('_c')
            def b(self):
                return 'b'
        a = A()
        self.assertEqual('b', a.b)
        self.assertEqual('b', a._c)

    def test_no_delete(self):

        def try_del(a):
            del a.b

        class A(object):

            @misc.cachedproperty
            def b(self):
                return 'b'
        a = A()
        self.assertEqual('b', a.b)
        self.assertRaises(AttributeError, try_del, a)
        self.assertEqual('b', a.b)

    def test_set(self):

        def try_set(a):
            a.b = 'c'

        class A(object):

            @misc.cachedproperty
            def b(self):
                return 'b'
        a = A()
        self.assertEqual('b', a.b)
        self.assertRaises(AttributeError, try_set, a)
        self.assertEqual('b', a.b)

    def test_documented_property(self):

        class A(object):

            @misc.cachedproperty
            def b(self):
                """I like bees."""
                return 'b'
        self.assertEqual('I like bees.', inspect.getdoc(A.b))

    def test_undocumented_property(self):

        class A(object):

            @misc.cachedproperty
            def b(self):
                return 'b'
        self.assertIsNone(inspect.getdoc(A.b))

    def test_threaded_access_property(self):
        called = collections.deque()

        class A(object):

            @misc.cachedproperty
            def b(self):
                called.append(1)
                time.sleep(random.random() * 0.5)
                return 'b'
        a = A()
        threads = []
        try:
            for _i in range(0, 20):
                t = threading_utils.daemon_thread(lambda: a.b)
                threads.append(t)
            for t in threads:
                t.start()
        finally:
            while threads:
                t = threads.pop()
                t.join()
        self.assertEqual(1, len(called))
        self.assertEqual('b', a.b)