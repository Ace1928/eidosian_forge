import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
def _write_output(self, output, out_name, out_mode):
    if out_name is not None:
        outfile = open(out_name, out_mode)
        try:
            outfile.write(output)
        finally:
            outfile.close()
        output = None
    return output