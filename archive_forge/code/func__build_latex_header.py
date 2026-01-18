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
@staticmethod
def _build_latex_header():
    latex_header = ['\\documentclass{article}', f'% !TeX program = {mpl.rcParams['pgf.texsystem']}', '\\usepackage{graphicx}', _get_preamble(), '\\begin{document}', '\\typeout{pgf_backend_query_start}']
    return '\n'.join(latex_header)