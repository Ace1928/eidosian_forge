import logging
import os
import sys
import warnings
from traitlets.config.loader import Config
from traitlets.config.application import boolean_flag, catch_config_error
from IPython.core import release
from IPython.core import usage
from IPython.core.completer import IPCompleter
from IPython.core.crashhandler import CrashHandler
from IPython.core.formatters import PlainTextFormatter
from IPython.core.history import HistoryManager
from IPython.core.application import (
from IPython.core.magic import MagicsManager
from IPython.core.magics import (
from IPython.core.shellapp import (
from IPython.extensions.storemagic import StoreMagics
from .interactiveshell import TerminalInteractiveShell
from IPython.paths import get_ipython_dir
from traitlets import (
class IPAppCrashHandler(CrashHandler):
    """sys.excepthook for IPython itself, leaves a detailed report on disk."""

    def __init__(self, app):
        contact_name = release.author
        contact_email = release.author_email
        bug_tracker = 'https://github.com/ipython/ipython/issues'
        super(IPAppCrashHandler, self).__init__(app, contact_name, contact_email, bug_tracker)

    def make_report(self, traceback):
        """Return a string containing a crash report."""
        sec_sep = self.section_sep
        report = [super(IPAppCrashHandler, self).make_report(traceback)]
        rpt_add = report.append
        try:
            rpt_add(sec_sep + 'History of session input:')
            for line in self.app.shell.user_ns['_ih']:
                rpt_add(line)
            rpt_add('\n*** Last line of input (may not be in above history):\n')
            rpt_add(self.app.shell._last_input_line + '\n')
        except:
            pass
        return ''.join(report)