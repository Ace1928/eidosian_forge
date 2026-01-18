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
def _escape_and_apply_props(s, prop):
    """
    Generate a TeX string that renders string *s* with font properties *prop*,
    also applying any required escapes to *s*.
    """
    commands = []
    families = {'serif': '\\rmfamily', 'sans': '\\sffamily', 'sans-serif': '\\sffamily', 'monospace': '\\ttfamily'}
    family = prop.get_family()[0]
    if family in families:
        commands.append(families[family])
    elif any((font.name == family for font in fm.fontManager.ttflist)) and mpl.rcParams['pgf.texsystem'] != 'pdflatex':
        commands.append('\\setmainfont{%s}\\rmfamily' % family)
    else:
        _log.warning('Ignoring unknown font: %s', family)
    size = prop.get_size_in_points()
    commands.append('\\fontsize{%f}{%f}' % (size, size * 1.2))
    styles = {'normal': '', 'italic': '\\itshape', 'oblique': '\\slshape'}
    commands.append(styles[prop.get_style()])
    boldstyles = ['semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black']
    if prop.get_weight() in boldstyles:
        commands.append('\\bfseries')
    commands.append('\\selectfont')
    return '{' + ''.join(commands) + '\\catcode`\\^=\\active\\def^{\\ifmmode\\sp\\else\\^{}\\fi}' + '\\catcode`\\%=\\active\\def%{\\%}' + _tex_escape(s) + '}'