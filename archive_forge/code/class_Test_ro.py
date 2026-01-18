import unittest
class Test_ro(unittest.TestCase):
    maxDiff = None

    def _callFUT(self, ob, **kwargs):
        from zope.interface.ro import _legacy_ro
        return _legacy_ro(ob, **kwargs)

    def test_w_empty_bases(self):

        class Foo:
            pass
        foo = Foo()
        foo.__bases__ = ()
        self.assertEqual(self._callFUT(foo), [foo])

    def test_w_single_base(self):

        class Foo:
            pass
        self.assertEqual(self._callFUT(Foo), [Foo, object])

    def test_w_bases(self):

        class Foo:
            pass

        class Bar(Foo):
            pass
        self.assertEqual(self._callFUT(Bar), [Bar, Foo, object])

    def test_w_diamond(self):

        class Foo:
            pass

        class Bar(Foo):
            pass

        class Baz(Foo):
            pass

        class Qux(Bar, Baz):
            pass
        self.assertEqual(self._callFUT(Qux), [Qux, Bar, Baz, Foo, object])

    def _make_IOErr(self):

        class Foo:

            def __init__(self, name, *bases):
                self.__name__ = name
                self.__bases__ = bases

            def __repr__(self):
                return self.__name__
        IEx = Foo('IEx')
        IStdErr = Foo('IStdErr', IEx)
        IEnvErr = Foo('IEnvErr', IStdErr)
        IIOErr = Foo('IIOErr', IEnvErr)
        IOSErr = Foo('IOSErr', IEnvErr)
        IOErr = Foo('IOErr', IEnvErr, IIOErr, IOSErr)
        return (IOErr, [IOErr, IIOErr, IOSErr, IEnvErr, IStdErr, IEx])

    def test_non_orderable(self):
        IOErr, bases = self._make_IOErr()
        self.assertEqual(self._callFUT(IOErr), bases)

    def test_mixed_inheritance_and_implementation(self):
        from zope.interface import Interface
        from zope.interface import implementedBy
        from zope.interface import implementer
        from zope.interface import providedBy

        class IFoo(Interface):
            pass

        @implementer(IFoo)
        class ImplementsFoo:
            pass

        class ExtendsFoo(ImplementsFoo):
            pass

        class ImplementsNothing:
            pass

        class ExtendsFooImplementsNothing(ExtendsFoo, ImplementsNothing):
            pass
        self.assertEqual(self._callFUT(providedBy(ExtendsFooImplementsNothing())), [implementedBy(ExtendsFooImplementsNothing), implementedBy(ExtendsFoo), implementedBy(ImplementsFoo), IFoo, Interface, implementedBy(ImplementsNothing), implementedBy(object)])