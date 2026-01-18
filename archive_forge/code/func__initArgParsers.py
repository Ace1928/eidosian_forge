import sys
import argparse
import os
import warnings
from . import loader, runner
from .signals import installHandler
def _initArgParsers(self):
    parent_parser = self._getParentArgParser()
    self._main_parser = self._getMainArgParser(parent_parser)
    self._discovery_parser = self._getDiscoveryArgParser(parent_parser)