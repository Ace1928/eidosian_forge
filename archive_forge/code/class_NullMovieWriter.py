import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import weakref
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.decorators import check_figures_equal
class NullMovieWriter(animation.AbstractMovieWriter):
    """
    A minimal MovieWriter.  It doesn't actually write anything.
    It just saves the arguments that were given to the setup() and
    grab_frame() methods as attributes, and counts how many times
    grab_frame() is called.

    This class doesn't have an __init__ method with the appropriate
    signature, and it doesn't define an isAvailable() method, so
    it cannot be added to the 'writers' registry.
    """

    def setup(self, fig, outfile, dpi, *args):
        self.fig = fig
        self.outfile = outfile
        self.dpi = dpi
        self.args = args
        self._count = 0

    def grab_frame(self, **savefig_kwargs):
        from matplotlib.animation import _validate_grabframe_kwargs
        _validate_grabframe_kwargs(savefig_kwargs)
        self.savefig_kwargs = savefig_kwargs
        self._count += 1

    def finish(self):
        pass