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
def _get_box_metrics(self, tex):
    """
        Get the width, total height and descent (in TeX points) for a TeX
        command's output in the current LaTeX environment.
        """
    self._stdin_writeln('{\\catcode`\\^=\\active\\catcode`\\%%=\\active\\sbox0{%s}\\typeout{\\the\\wd0,\\the\\ht0,\\the\\dp0}}' % tex)
    try:
        answer = self._expect_prompt()
    except LatexError as err:
        raise ValueError('Error measuring {}\nLaTeX Output:\n{}'.format(tex, err.latex_output)) from err
    try:
        width, height, offset = answer.splitlines()[-3].split(',')
    except Exception as err:
        raise ValueError('Error measuring {}\nLaTeX Output:\n{}'.format(tex, answer)) from err
    w, h, o = (float(width[:-2]), float(height[:-2]), float(offset[:-2]))
    return (w, h + o, o)