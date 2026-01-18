import sys
import argparse
import os
import warnings
from . import loader, runner
from .signals import installHandler
def _getParentArgParser(self):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-v', '--verbose', dest='verbosity', action='store_const', const=2, help='Verbose output')
    parser.add_argument('-q', '--quiet', dest='verbosity', action='store_const', const=0, help='Quiet output')
    parser.add_argument('--locals', dest='tb_locals', action='store_true', help='Show local variables in tracebacks')
    if self.failfast is None:
        parser.add_argument('-f', '--failfast', dest='failfast', action='store_true', help='Stop on first fail or error')
        self.failfast = False
    if self.catchbreak is None:
        parser.add_argument('-c', '--catch', dest='catchbreak', action='store_true', help='Catch Ctrl-C and display results so far')
        self.catchbreak = False
    if self.buffer is None:
        parser.add_argument('-b', '--buffer', dest='buffer', action='store_true', help='Buffer stdout and stderr during tests')
        self.buffer = False
    if self.testNamePatterns is None:
        parser.add_argument('-k', dest='testNamePatterns', action='append', type=_convert_select_pattern, help='Only run tests which match the given substring')
        self.testNamePatterns = []
    return parser