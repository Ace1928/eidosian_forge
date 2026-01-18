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
class LatexManager:
    """
    The LatexManager opens an instance of the LaTeX application for
    determining the metrics of text elements. The LaTeX environment can be
    modified by setting fonts and/or a custom preamble in `.rcParams`.
    """

    @staticmethod
    def _build_latex_header():
        latex_header = ['\\documentclass{article}', f'% !TeX program = {mpl.rcParams['pgf.texsystem']}', '\\usepackage{graphicx}', _get_preamble(), '\\begin{document}', '\\typeout{pgf_backend_query_start}']
        return '\n'.join(latex_header)

    @classmethod
    def _get_cached_or_new(cls):
        """
        Return the previous LatexManager if the header and tex system did not
        change, or a new instance otherwise.
        """
        return cls._get_cached_or_new_impl(cls._build_latex_header())

    @classmethod
    @functools.lru_cache(1)
    def _get_cached_or_new_impl(cls, header):
        return cls()

    def _stdin_writeln(self, s):
        if self.latex is None:
            self._setup_latex_process()
        self.latex.stdin.write(s)
        self.latex.stdin.write('\n')
        self.latex.stdin.flush()

    def _expect(self, s):
        s = list(s)
        chars = []
        while True:
            c = self.latex.stdout.read(1)
            chars.append(c)
            if chars[-len(s):] == s:
                break
            if not c:
                self.latex.kill()
                self.latex = None
                raise LatexError('LaTeX process halted', ''.join(chars))
        return ''.join(chars)

    def _expect_prompt(self):
        return self._expect('\n*')

    def __init__(self):
        self._tmpdir = TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self._finalize_tmpdir = weakref.finalize(self, self._tmpdir.cleanup)
        self._setup_latex_process(expect_reply=False)
        stdout, stderr = self.latex.communicate('\n\\makeatletter\\@@end\n')
        if self.latex.returncode != 0:
            raise LatexError(f'LaTeX errored (probably missing font or error in preamble) while processing the following input:\n{self._build_latex_header()}', stdout)
        self.latex = None
        self._get_box_metrics = functools.lru_cache(self._get_box_metrics)

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

    def get_width_height_descent(self, text, prop):
        """
        Get the width, total height, and descent (in TeX points) for a text
        typeset by the current LaTeX environment.
        """
        return self._get_box_metrics(_escape_and_apply_props(text, prop))

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