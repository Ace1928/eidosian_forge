import sys
import argparse
import os
import warnings
from . import loader, runner
from .signals import installHandler
def _getMainArgParser(self, parent):
    parser = argparse.ArgumentParser(parents=[parent])
    parser.prog = self.progName
    parser.print_help = self._print_help
    parser.add_argument('tests', nargs='*', help='a list of any number of test modules, classes and test methods.')
    return parser