import sys
import unittest
class Test_asStructuredText(unittest.TestCase):

    def _callFUT(self, iface):
        from zope.interface.document import asStructuredText
        return asStructuredText(iface)

    def test_asStructuredText_no_docstring(self):
        from zope.interface import Interface
        EXPECTED = '\n\n'.join(['INoDocstring', ' Attributes:', ' Methods:', ''])

        class INoDocstring(Interface):
            pass
        self.assertEqual(self._callFUT(INoDocstring), EXPECTED)

    def test_asStructuredText_empty_with_docstring(self):
        from zope.interface import Interface
        EXPECTED = '\n\n'.join(['IEmpty', ' This is an empty interface.', ' Attributes:', ' Methods:', ''])

        class IEmpty(Interface):
            """ This is an empty interface.
            """
        self.assertEqual(self._callFUT(IEmpty), EXPECTED)

    def test_asStructuredText_empty_with_multiline_docstring(self):
        from zope.interface import Interface
        indent = ' ' * 12 if sys.version_info < (3, 13) else ''
        EXPECTED = '\n'.join(['IEmpty', '', ' This is an empty interface.', ' ', f'{indent} It can be used to annotate any class or object, because it promises', f'{indent} nothing.', '', ' Attributes:', '', ' Methods:', '', ''])

        class IEmpty(Interface):
            """ This is an empty interface.

            It can be used to annotate any class or object, because it promises
            nothing.
            """
        self.assertEqual(self._callFUT(IEmpty), EXPECTED)

    def test_asStructuredText_with_attribute_no_docstring(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        EXPECTED = '\n\n'.join(['IHasAttribute', ' This interface has an attribute.', ' Attributes:', '  an_attribute -- no documentation', ' Methods:', ''])

        class IHasAttribute(Interface):
            """ This interface has an attribute.
            """
            an_attribute = Attribute('an_attribute')
        self.assertEqual(self._callFUT(IHasAttribute), EXPECTED)

    def test_asStructuredText_with_attribute_with_docstring(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        EXPECTED = '\n\n'.join(['IHasAttribute', ' This interface has an attribute.', ' Attributes:', '  an_attribute -- This attribute is documented.', ' Methods:', ''])

        class IHasAttribute(Interface):
            """ This interface has an attribute.
            """
            an_attribute = Attribute('an_attribute', 'This attribute is documented.')
        self.assertEqual(self._callFUT(IHasAttribute), EXPECTED)

    def test_asStructuredText_with_method_no_args_no_docstring(self):
        from zope.interface import Interface
        EXPECTED = '\n\n'.join(['IHasMethod', ' This interface has a method.', ' Attributes:', ' Methods:', '  aMethod() -- no documentation', ''])

        class IHasMethod(Interface):
            """ This interface has a method.
            """

            def aMethod():
                pass
        self.assertEqual(self._callFUT(IHasMethod), EXPECTED)

    def test_asStructuredText_with_method_positional_args_no_docstring(self):
        from zope.interface import Interface
        EXPECTED = '\n\n'.join(['IHasMethod', ' This interface has a method.', ' Attributes:', ' Methods:', '  aMethod(first, second) -- no documentation', ''])

        class IHasMethod(Interface):
            """ This interface has a method.
            """

            def aMethod(first, second):
                pass
        self.assertEqual(self._callFUT(IHasMethod), EXPECTED)

    def test_asStructuredText_with_method_starargs_no_docstring(self):
        from zope.interface import Interface
        EXPECTED = '\n\n'.join(['IHasMethod', ' This interface has a method.', ' Attributes:', ' Methods:', '  aMethod(first, second, *rest) -- no documentation', ''])

        class IHasMethod(Interface):
            """ This interface has a method.
            """

            def aMethod(first, second, *rest):
                pass
        self.assertEqual(self._callFUT(IHasMethod), EXPECTED)

    def test_asStructuredText_with_method_kwargs_no_docstring(self):
        from zope.interface import Interface
        EXPECTED = '\n\n'.join(['IHasMethod', ' This interface has a method.', ' Attributes:', ' Methods:', '  aMethod(first, second, **kw) -- no documentation', ''])

        class IHasMethod(Interface):
            """ This interface has a method.
            """

            def aMethod(first, second, **kw):
                pass
        self.assertEqual(self._callFUT(IHasMethod), EXPECTED)

    def test_asStructuredText_with_method_with_docstring(self):
        from zope.interface import Interface
        EXPECTED = '\n\n'.join(['IHasMethod', ' This interface has a method.', ' Attributes:', ' Methods:', '  aMethod() -- This method is documented.', ''])

        class IHasMethod(Interface):
            """ This interface has a method.
            """

            def aMethod():
                """This method is documented.
                """
        self.assertEqual(self._callFUT(IHasMethod), EXPECTED)

    def test_asStructuredText_derived_ignores_base(self):
        from zope.interface import Attribute
        from zope.interface import Interface
        EXPECTED = '\n\n'.join(['IDerived', ' IDerived doc', ' This interface extends:', '  o IBase', ' Attributes:', '  attr1 -- no documentation', '  attr2 -- attr2 doc', ' Methods:', '  method3() -- method3 doc', '  method4() -- no documentation', '  method5() -- method5 doc', ''])

        class IBase(Interface):

            def method1():
                """docstring"""

            def method2():
                """docstring"""

        class IDerived(IBase):
            """IDerived doc"""
            attr1 = Attribute('attr1')
            attr2 = Attribute('attr2', 'attr2 doc')

            def method3():
                """method3 doc"""

            def method4():
                pass

            def method5():
                """method5 doc"""
        self.assertEqual(self._callFUT(IDerived), EXPECTED)