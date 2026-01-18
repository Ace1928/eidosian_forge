import sys
import unittest
import sys
class Test_isClassAdvisor(unittest.TestCase):

    def _callFUT(self, *args, **kw):
        from zope.interface.advice import isClassAdvisor
        return isClassAdvisor(*args, **kw)

    def test_w_non_function(self):
        self.assertEqual(self._callFUT(self), False)

    def test_w_normal_function(self):

        def foo():
            raise NotImplementedError()
        self.assertEqual(self._callFUT(foo), False)

    def test_w_advisor_function(self):

        def bar():
            raise NotImplementedError()
        bar.previousMetaclass = object()
        self.assertEqual(self._callFUT(bar), True)