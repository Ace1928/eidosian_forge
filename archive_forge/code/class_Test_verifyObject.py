import unittest
class Test_verifyObject(Test_verifyClass):

    @classmethod
    def _get_FUT(cls):
        from zope.interface.verify import verifyObject
        return verifyObject

    def _adjust_object_before_verify(self, target):
        if isinstance(target, (type, type(OldSkool))):
            target = target()
        return target

    def test_class_misses_attribute_for_attribute(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        from zope.interface import implementer
        from zope.interface.exceptions import BrokenImplementation

        class ICurrent(Interface):
            attr = Attribute('The foo Attribute')

        @implementer(ICurrent)
        class Current:
            pass
        self.assertRaises(BrokenImplementation, self._callFUT, ICurrent, Current)

    def test_module_hit(self):
        from zope.interface.tests import dummy
        from zope.interface.tests.idummy import IDummyModule
        self._callFUT(IDummyModule, dummy)

    def test_module_miss(self):
        from zope.interface import Interface
        from zope.interface.exceptions import DoesNotImplement
        from zope.interface.tests import dummy

        class IDummyModule(Interface):
            pass
        self.assertRaises(DoesNotImplement, self._callFUT, IDummyModule, dummy)

    def test_staticmethod_hit_on_class(self):
        from zope.interface import Interface
        from zope.interface import provider
        from zope.interface.verify import verifyObject

        class IFoo(Interface):

            def bar(a, b):
                """The bar method"""

        @provider(IFoo)
        class Foo:

            @staticmethod
            def bar(a, b):
                raise AssertionError("We're never actually called")
        verifyObject(IFoo, Foo)