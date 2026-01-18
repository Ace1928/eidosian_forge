import sys, os
import textwrap
def _add_version_option(self):
    self.add_option('--version', action='version', help=_("show program's version number and exit"))