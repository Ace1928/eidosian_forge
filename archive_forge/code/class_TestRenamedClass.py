import sys
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
from pyomo.common.log import LoggingIntercept
from io import StringIO
import logging
class TestRenamedClass(unittest.TestCase):

    def test_renamed(self):

        class NewClass(object):
            attr = 'NewClass'

        class NewClassSubclass(NewClass):
            pass
        out = StringIO()
        with LoggingIntercept(out):

            class DeprecatedClass(metaclass=RenamedClass):
                __renamed__new_class__ = NewClass
                __renamed__version__ = 'X.y'
        self.assertEqual(out.getvalue(), '')
        out = StringIO()
        with LoggingIntercept(out):

            class DeprecatedClassSubclass(DeprecatedClass):
                attr = 'DeprecatedClassSubclass'
        self.assertRegex(out.getvalue().replace('\n', ' ').strip(), "^DEPRECATED: Declaring class 'DeprecatedClassSubclass' derived from 'DeprecatedClass'.  The class 'DeprecatedClass' has been renamed to 'NewClass'.  \\(deprecated in X.y\\) \\(called from [^\\)]*\\)$")
        out = StringIO()
        with LoggingIntercept(out):

            class DeprecatedClassSubSubclass(DeprecatedClassSubclass):
                attr = 'DeprecatedClassSubSubclass'
        self.assertEqual(out.getvalue(), '')
        out = StringIO()
        with LoggingIntercept(out):
            newclass = NewClass()
            newclasssubclass = NewClassSubclass()
        self.assertEqual(out.getvalue(), '')
        out = StringIO()
        with LoggingIntercept(out):
            deprecatedclass = DeprecatedClass()
        self.assertRegex(out.getvalue().replace('\n', ' ').strip(), "^DEPRECATED: Instantiating class 'DeprecatedClass'.  The class 'DeprecatedClass' has been renamed to 'NewClass'.  \\(deprecated in X.y\\) \\(called from [^\\)]*\\)$")
        out = StringIO()
        with LoggingIntercept(out):
            deprecatedsubclass = DeprecatedClassSubclass()
            deprecatedsubsubclass = DeprecatedClassSubSubclass()
        self.assertEqual(out.getvalue(), '')
        out = StringIO()
        with LoggingIntercept(out):
            self.assertIsInstance(deprecatedsubclass, NewClass)
            self.assertIsInstance(deprecatedsubsubclass, NewClass)
        self.assertEqual(out.getvalue(), '')
        for obj in (newclass, newclasssubclass, deprecatedclass, deprecatedsubclass, deprecatedsubsubclass):
            out = StringIO()
            with LoggingIntercept(out):
                self.assertIsInstance(obj, DeprecatedClass)
            self.assertRegex(out.getvalue().replace('\n', ' ').strip(), "^DEPRECATED: Checking type relative to 'DeprecatedClass'.  The class 'DeprecatedClass' has been renamed to 'NewClass'.  \\(deprecated in X.y\\) \\(called from [^\\)]*\\)$")
        out = StringIO()
        with LoggingIntercept(out):
            self.assertTrue(issubclass(DeprecatedClass, NewClass))
            self.assertTrue(issubclass(DeprecatedClassSubclass, NewClass))
            self.assertTrue(issubclass(DeprecatedClassSubSubclass, NewClass))
        self.assertEqual(out.getvalue(), '')
        for cls in (NewClass, NewClassSubclass, DeprecatedClass, DeprecatedClassSubclass, DeprecatedClassSubSubclass):
            out = StringIO()
            with LoggingIntercept(out):
                self.assertTrue(issubclass(cls, DeprecatedClass))
            self.assertRegex(out.getvalue().replace('\n', ' ').strip(), "^DEPRECATED: Checking type relative to 'DeprecatedClass'.  The class 'DeprecatedClass' has been renamed to 'NewClass'.  \\(deprecated in X.y\\) \\(called from [^\\)]*\\)$")
        self.assertEqual(newclass.attr, 'NewClass')
        self.assertEqual(newclasssubclass.attr, 'NewClass')
        self.assertEqual(deprecatedclass.attr, 'NewClass')
        self.assertEqual(deprecatedsubclass.attr, 'DeprecatedClassSubclass')
        self.assertEqual(deprecatedsubsubclass.attr, 'DeprecatedClassSubSubclass')
        self.assertEqual(NewClass.attr, 'NewClass')
        self.assertEqual(NewClassSubclass.attr, 'NewClass')
        self.assertEqual(DeprecatedClass.attr, 'NewClass')
        self.assertEqual(DeprecatedClassSubclass.attr, 'DeprecatedClassSubclass')
        self.assertEqual(DeprecatedClassSubSubclass.attr, 'DeprecatedClassSubSubclass')

    def test_renamed_errors(self):

        class NewClass(object):
            pass
        with self.assertRaisesRegex(TypeError, "Declaring class 'DeprecatedClass' using the RenamedClass metaclass, but without specifying the __renamed__new_class__ class attribute"):

            class DeprecatedClass(metaclass=RenamedClass):
                __renamed_new_class__ = NewClass
        with self.assertRaisesRegex(DeveloperError, "Declaring class 'DeprecatedClass' using the RenamedClass metaclass, but without specifying the __renamed__version__ class attribute", normalize_whitespace=True):

            class DeprecatedClass(metaclass=RenamedClass):
                __renamed__new_class__ = NewClass