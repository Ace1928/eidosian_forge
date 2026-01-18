from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
class TestIncompatiblePrintOperator(TestCase):
    """
    Tests for warning about invalid use of print function.
    """

    def test_valid_print(self):
        self.flakes('\n        print("Hello")\n        ')

    def test_invalid_print_when_imported_from_future(self):
        exc = self.flakes('\n        from __future__ import print_function\n        import sys\n        print >>sys.stderr, "Hello"\n        ', m.InvalidPrintSyntax).messages[0]
        self.assertEqual(exc.lineno, 4)
        self.assertEqual(exc.col, 0)

    def test_print_augmented_assign(self):
        self.flakes('print += 1')

    def test_print_function_assignment(self):
        """
        A valid assignment, tested for catching false positives.
        """
        self.flakes('\n        from __future__ import print_function\n        log = print\n        log("Hello")\n        ')

    def test_print_in_lambda(self):
        self.flakes('\n        from __future__ import print_function\n        a = lambda: print\n        ')

    def test_print_returned_in_function(self):
        self.flakes('\n        from __future__ import print_function\n        def a():\n            return print\n        ')

    def test_print_as_condition_test(self):
        self.flakes('\n        from __future__ import print_function\n        if print: pass\n        ')