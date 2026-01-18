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
def _setup_latex_process(self, *, expect_reply=True):
    try:
        self.latex = subprocess.Popen([mpl.rcParams['pgf.texsystem'], '-halt-on-error'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf-8', cwd=self.tmpdir)
    except FileNotFoundError as err:
        raise RuntimeError(f"{mpl.rcParams['pgf.texsystem']!r} not found; install it or change rcParams['pgf.texsystem'] to an available TeX implementation") from err
    except OSError as err:
        raise RuntimeError(f'Error starting {mpl.rcParams['pgf.texsystem']!r}') from err

    def finalize_latex(latex):
        latex.kill()
        try:
            latex.communicate()
        except RuntimeError:
            latex.wait()
    self._finalize_latex = weakref.finalize(self, finalize_latex, self.latex)
    self._stdin_writeln(self._build_latex_header())
    if expect_reply:
        self._expect('*pgf_backend_query_start')
        self._expect_prompt()