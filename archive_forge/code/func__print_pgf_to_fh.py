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
def _print_pgf_to_fh(self, fh, *, bbox_inches_restore=None):
    header_text = '%% Creator: Matplotlib, PGF backend\n%%\n%% To include the figure in your LaTeX document, write\n%%   \\input{<filename>.pgf}\n%%\n%% Make sure the required packages are loaded in your preamble\n%%   \\usepackage{pgf}\n%%\n%% Also ensure that all the required font packages are loaded; for instance,\n%% the lmodern package is sometimes necessary when using math font.\n%%   \\usepackage{lmodern}\n%%\n%% Figures using additional raster images can only be included by \\input if\n%% they are in the same directory as the main LaTeX file. For loading figures\n%% from other directories you can use the `import` package\n%%   \\usepackage{import}\n%%\n%% and then include the figures with\n%%   \\import{<path to file>}{<filename>.pgf}\n%%\n'
    header_info_preamble = ['%% Matplotlib used the following preamble']
    for line in _get_preamble().splitlines():
        header_info_preamble.append('%%   ' + line)
    header_info_preamble.append('%%')
    header_info_preamble = '\n'.join(header_info_preamble)
    w, h = (self.figure.get_figwidth(), self.figure.get_figheight())
    dpi = self.figure.dpi
    fh.write(header_text)
    fh.write(header_info_preamble)
    fh.write('\n')
    _writeln(fh, '\\begingroup')
    _writeln(fh, '\\makeatletter')
    _writeln(fh, '\\begin{pgfpicture}')
    _writeln(fh, '\\pgfpathrectangle{\\pgfpointorigin}{\\pgfqpoint{%fin}{%fin}}' % (w, h))
    _writeln(fh, '\\pgfusepath{use as bounding box, clip}')
    renderer = MixedModeRenderer(self.figure, w, h, dpi, RendererPgf(self.figure, fh), bbox_inches_restore=bbox_inches_restore)
    self.figure.draw(renderer)
    _writeln(fh, '\\end{pgfpicture}')
    _writeln(fh, '\\makeatother')
    _writeln(fh, '\\endgroup')