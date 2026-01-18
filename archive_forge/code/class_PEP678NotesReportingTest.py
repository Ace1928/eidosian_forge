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
class PEP678NotesReportingTest(unittest.TestCase):
    ERROR_WITH_NOTE = '\ntry:\n    raise AssertionError("Message")\nexcept Exception as e:\n    try:\n        e.add_note("This is a PEP-678 note.")\n    except AttributeError:  # Python <= 3.10\n        e.__notes__ = ("This is a PEP-678 note.",)\n    raise\n    '

    def test_verbose_reports_notes(self):
        with tt.AssertPrints(['AssertionError', 'Message', 'This is a PEP-678 note.']):
            ip.run_cell(self.ERROR_WITH_NOTE)

    def test_plain_reports_notes(self):
        with tt.AssertPrints(['AssertionError', 'Message', 'This is a PEP-678 note.']):
            ip.run_cell('%xmode Plain')
            ip.run_cell(self.ERROR_WITH_NOTE)
            ip.run_cell('%xmode Verbose')