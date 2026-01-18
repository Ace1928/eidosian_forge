import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
class TestImports(_DoctestMixin, TestImports):
    """Run TestImports with each test wrapped in a doctest."""