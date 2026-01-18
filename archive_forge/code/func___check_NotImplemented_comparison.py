import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def __check_NotImplemented_comparison(self, name):
    import operator
    ib = self._makeOneToCompare()
    op = getattr(operator, name)
    meth = getattr(ib, '__%s__' % name)

    class RaisesErrorOnMissing:
        Exc = AttributeError

        def __getattribute__(self, name):
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                exc = RaisesErrorOnMissing.Exc
                raise exc(name)

    class RaisesErrorOnModule(RaisesErrorOnMissing):

        def __init__(self):
            self.__name__ = 'foo'

        @property
        def __module__(self):
            raise AttributeError

    class RaisesErrorOnName(RaisesErrorOnMissing):

        def __init__(self):
            self.__module__ = 'foo'
    self.assertEqual(RaisesErrorOnModule().__name__, 'foo')
    self.assertEqual(RaisesErrorOnName().__module__, 'foo')
    with self.assertRaises(AttributeError):
        getattr(RaisesErrorOnModule(), '__module__')
    with self.assertRaises(AttributeError):
        getattr(RaisesErrorOnName(), '__name__')
    for cls in (RaisesErrorOnModule, RaisesErrorOnName):
        self.assertIs(meth(cls()), NotImplemented)

    class AllowsAnyComparison(RaisesErrorOnMissing):

        def __eq__(self, other):
            return True
        __lt__ = __eq__
        __le__ = __eq__
        __gt__ = __eq__
        __ge__ = __eq__
        __ne__ = __eq__
    self.assertTrue(op(ib, AllowsAnyComparison()))
    self.assertIs(meth(AllowsAnyComparison()), NotImplemented)

    class AllowsNoComparison:
        __eq__ = None
        __lt__ = __eq__
        __le__ = __eq__
        __gt__ = __eq__
        __ge__ = __eq__
        __ne__ = __eq__
    self.assertIs(meth(AllowsNoComparison()), NotImplemented)
    with self.assertRaises(TypeError):
        op(ib, AllowsNoComparison())

    class MyException(Exception):
        pass
    RaisesErrorOnMissing.Exc = MyException
    with self.assertRaises(MyException):
        getattr(RaisesErrorOnModule(), '__module__')
    with self.assertRaises(MyException):
        getattr(RaisesErrorOnName(), '__name__')
    for cls in (RaisesErrorOnModule, RaisesErrorOnName):
        with self.assertRaises(MyException):
            op(ib, cls())
        with self.assertRaises(MyException):
            meth(cls())