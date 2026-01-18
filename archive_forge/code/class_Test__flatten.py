import unittest
class Test__flatten(unittest.TestCase):

    def _callFUT(self, ob):
        from zope.interface.ro import _legacy_flatten
        return _legacy_flatten(ob)

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
        self.assertEqual(self._callFUT(Qux), [Qux, Bar, Foo, object, Baz, Foo, object])