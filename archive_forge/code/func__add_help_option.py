import sys, os
import textwrap
def _add_help_option(self):
    self.add_option('-h', '--help', action='help', help=_('show this help message and exit'))