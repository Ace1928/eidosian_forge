import io
import os.path
import platform
import re
import sys
import traceback
import unittest
from textwrap import dedent
from tempfile import TemporaryDirectory
from IPython.core.ultratb import ColorTB, VerboseTB
from IPython.testing import tools as tt
from IPython.testing.decorators import onlyif_unicode_paths, skip_without
from IPython.utils.syspathcontext import prepended_to_syspath
import sys
class Python3ChainedExceptionsTest(unittest.TestCase):
    DIRECT_CAUSE_ERROR_CODE = "\ntry:\n    x = 1 + 2\n    print(not_defined_here)\nexcept Exception as e:\n    x += 55\n    x - 1\n    y = {}\n    raise KeyError('uh') from e\n    "
    EXCEPTION_DURING_HANDLING_CODE = "\ntry:\n    x = 1 + 2\n    print(not_defined_here)\nexcept Exception as e:\n    x += 55\n    x - 1\n    y = {}\n    raise KeyError('uh')\n    "
    SUPPRESS_CHAINING_CODE = '\ntry:\n    1/0\nexcept Exception:\n    raise ValueError("Yikes") from None\n    '

    def test_direct_cause_error(self):
        with tt.AssertPrints(['KeyError', 'NameError', 'direct cause']):
            ip.run_cell(self.DIRECT_CAUSE_ERROR_CODE)

    def test_exception_during_handling_error(self):
        with tt.AssertPrints(['KeyError', 'NameError', 'During handling']):
            ip.run_cell(self.EXCEPTION_DURING_HANDLING_CODE)

    def test_suppress_exception_chaining(self):
        with tt.AssertNotPrints('ZeroDivisionError'), tt.AssertPrints('ValueError', suppress=False):
            ip.run_cell(self.SUPPRESS_CHAINING_CODE)

    def test_plain_direct_cause_error(self):
        with tt.AssertPrints(['KeyError', 'NameError', 'direct cause']):
            ip.run_cell('%xmode Plain')
            ip.run_cell(self.DIRECT_CAUSE_ERROR_CODE)
            ip.run_cell('%xmode Verbose')

    def test_plain_exception_during_handling_error(self):
        with tt.AssertPrints(['KeyError', 'NameError', 'During handling']):
            ip.run_cell('%xmode Plain')
            ip.run_cell(self.EXCEPTION_DURING_HANDLING_CODE)
            ip.run_cell('%xmode Verbose')

    def test_plain_suppress_exception_chaining(self):
        with tt.AssertNotPrints('ZeroDivisionError'), tt.AssertPrints('ValueError', suppress=False):
            ip.run_cell('%xmode Plain')
            ip.run_cell(self.SUPPRESS_CHAINING_CODE)
            ip.run_cell('%xmode Verbose')