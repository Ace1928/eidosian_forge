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
class FigureCanvasPgf(FigureCanvasBase):
    filetypes = {'pgf': 'LaTeX PGF picture', 'pdf': 'LaTeX compiled PGF picture', 'png': 'Portable Network Graphics'}

    def get_default_filetype(self):
        return 'pdf'

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

    def print_pgf(self, fname_or_fh, **kwargs):
        """
        Output pgf macros for drawing the figure so it can be included and
        rendered in latex documents.
        """
        with cbook.open_file_cm(fname_or_fh, 'w', encoding='utf-8') as file:
            if not cbook.file_requires_unicode(file):
                file = codecs.getwriter('utf-8')(file)
            self._print_pgf_to_fh(file, **kwargs)

    def print_pdf(self, fname_or_fh, *, metadata=None, **kwargs):
        """Use LaTeX to compile a pgf generated figure to pdf."""
        w, h = self.figure.get_size_inches()
        info_dict = _create_pdf_info_dict('pgf', metadata or {})
        pdfinfo = ','.join((_metadata_to_str(k, v) for k, v in info_dict.items()))
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir)
            self.print_pgf(tmppath / 'figure.pgf', **kwargs)
            (tmppath / 'figure.tex').write_text('\n'.join(['\\documentclass[12pt]{article}', '\\usepackage[pdfinfo={%s}]{hyperref}' % pdfinfo, '\\usepackage[papersize={%fin,%fin}, margin=0in]{geometry}' % (w, h), '\\usepackage{pgf}', _get_preamble(), '\\begin{document}', '\\centering', '\\input{figure.pgf}', '\\end{document}']), encoding='utf-8')
            texcommand = mpl.rcParams['pgf.texsystem']
            cbook._check_and_log_subprocess([texcommand, '-interaction=nonstopmode', '-halt-on-error', 'figure.tex'], _log, cwd=tmpdir)
            with (tmppath / 'figure.pdf').open('rb') as orig, cbook.open_file_cm(fname_or_fh, 'wb') as dest:
                shutil.copyfileobj(orig, dest)

    def print_png(self, fname_or_fh, **kwargs):
        """Use LaTeX to compile a pgf figure to pdf and convert it to png."""
        converter = make_pdf_to_png_converter()
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir)
            pdf_path = tmppath / 'figure.pdf'
            png_path = tmppath / 'figure.png'
            self.print_pdf(pdf_path, **kwargs)
            converter(pdf_path, png_path, dpi=self.figure.dpi)
            with png_path.open('rb') as orig, cbook.open_file_cm(fname_or_fh, 'wb') as dest:
                shutil.copyfileobj(orig, dest)

    def get_renderer(self):
        return RendererPgf(self.figure, None)

    def draw(self):
        self.figure.draw_without_rendering()
        return super().draw()