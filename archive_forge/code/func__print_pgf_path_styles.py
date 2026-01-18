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
def _print_pgf_path_styles(self, gc, rgbFace):
    capstyles = {'butt': '\\pgfsetbuttcap', 'round': '\\pgfsetroundcap', 'projecting': '\\pgfsetrectcap'}
    _writeln(self.fh, capstyles[gc.get_capstyle()])
    joinstyles = {'miter': '\\pgfsetmiterjoin', 'round': '\\pgfsetroundjoin', 'bevel': '\\pgfsetbeveljoin'}
    _writeln(self.fh, joinstyles[gc.get_joinstyle()])
    has_fill = rgbFace is not None
    if gc.get_forced_alpha():
        fillopacity = strokeopacity = gc.get_alpha()
    else:
        strokeopacity = gc.get_rgb()[3]
        fillopacity = rgbFace[3] if has_fill and len(rgbFace) > 3 else 1.0
    if has_fill:
        _writeln(self.fh, '\\definecolor{currentfill}{rgb}{%f,%f,%f}' % tuple(rgbFace[:3]))
        _writeln(self.fh, '\\pgfsetfillcolor{currentfill}')
    if has_fill and fillopacity != 1.0:
        _writeln(self.fh, '\\pgfsetfillopacity{%f}' % fillopacity)
    lw = gc.get_linewidth() * mpl_pt_to_in * latex_in_to_pt
    stroke_rgba = gc.get_rgb()
    _writeln(self.fh, '\\pgfsetlinewidth{%fpt}' % lw)
    _writeln(self.fh, '\\definecolor{currentstroke}{rgb}{%f,%f,%f}' % stroke_rgba[:3])
    _writeln(self.fh, '\\pgfsetstrokecolor{currentstroke}')
    if strokeopacity != 1.0:
        _writeln(self.fh, '\\pgfsetstrokeopacity{%f}' % strokeopacity)
    dash_offset, dash_list = gc.get_dashes()
    if dash_list is None:
        _writeln(self.fh, '\\pgfsetdash{}{0pt}')
    else:
        _writeln(self.fh, '\\pgfsetdash{%s}{%fpt}' % (''.join(('{%fpt}' % dash for dash in dash_list)), dash_offset))