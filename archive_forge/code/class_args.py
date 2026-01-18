import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
class args(object):

    def __init__(self, max_width=0, print_empty=False, fit_width=False):
        self.fit_width = fit_width
        if max_width > 0:
            self.max_width = max_width
        else:
            self.max_width = int(os.environ.get('CLIFF_MAX_TERM_WIDTH', 0))
        self.print_empty = print_empty