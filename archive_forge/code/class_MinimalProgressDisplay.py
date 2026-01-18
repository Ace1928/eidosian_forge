import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
class MinimalProgressDisplay:
    """A minimalist replacement for tqdm.tqdm"""

    def __init__(self, total):
        self.count = 0
        self.total = total

    def __repr__(self):
        """represent current completion"""
        return str(self.count) + '/' + str(self.total)

    def render(self):
        """print self.__repr__ to stderr"""
        print(f'\r{self}', file=sys.stderr, end='')

    def update(self, i):
        """modify completion and render"""
        self.count = i
        self.render()

    def reset(self):
        """set counter to 0"""
        self.count = 0

    @staticmethod
    def close():
        """print a new empty line"""
        print('', file=sys.stderr)