import fnmatch
import os
import re
import sys
import pkg_resources
from . import copydir
from . import pluginlib
from .command import Command, BadCommand
def _show_template_vars(self, tmpl_name, tmpl, message=None):
    title = '%s (from %s)' % (tmpl.name, tmpl_name)
    print(title)
    print('-' * len(title))
    if message is not None:
        print('  %s' % message)
        print()
        return
    tmpl.print_vars(indent=2)