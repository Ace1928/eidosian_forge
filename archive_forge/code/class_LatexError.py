import codecs
import datetime
import functools
from io import BytesIO
import logging
import math
import os
import pathlib
import shutil
import subprocess
from tempfile import TemporaryDirectory
import weakref
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_pdf import (
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib._pylab_helpers import Gcf
class LatexError(Exception):

    def __init__(self, message, latex_output=''):
        super().__init__(message)
        self.latex_output = latex_output

    def __str__(self):
        s, = self.args
        if self.latex_output:
            s += '\n' + self.latex_output
        return s