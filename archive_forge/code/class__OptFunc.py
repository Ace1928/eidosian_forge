import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
class _OptFunc:

    def __init__(self, opt):
        self.opt = opt

    def __call__(self, *args):
        """Return stored dict."""
        return self.opt