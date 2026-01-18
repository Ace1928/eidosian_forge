import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import (
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps
def _print_figure_tex(self, fmt, outfile, *, dpi, dsc_comments, orientation, papertype, bbox_inches_restore=None):
    """
        If :rc:`text.usetex` is True, a temporary pair of tex/eps files
        are created to allow tex to manage the text layout via the PSFrags
        package. These files are processed to yield the final ps or eps file.

        The rest of the behavior is as for `._print_figure`.
        """
    is_eps = fmt == 'eps'
    width, height = self.figure.get_size_inches()
    xo = 0
    yo = 0
    llx = xo
    lly = yo
    urx = llx + self.figure.bbox.width
    ury = lly + self.figure.bbox.height
    bbox = (llx, lly, urx, ury)
    self._pswriter = StringIO()
    ps_renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)
    renderer = MixedModeRenderer(self.figure, width, height, dpi, ps_renderer, bbox_inches_restore=bbox_inches_restore)
    self.figure.draw(renderer)
    with TemporaryDirectory() as tmpdir:
        tmppath = pathlib.Path(tmpdir, 'tmp.ps')
        tmppath.write_text(f'%!PS-Adobe-3.0 EPSF-3.0\n%%LanguageLevel: 3\n{dsc_comments}\n{get_bbox_header(bbox)[0]}\n%%EndComments\n%%BeginProlog\n/mpldict {len(_psDefs)} dict def\nmpldict begin\n{''.join(_psDefs)}\nend\n%%EndProlog\nmpldict begin\n{_nums_to_str(xo, yo)} translate\n0 0 {_nums_to_str(width * 72, height * 72)} rectclip\n{self._pswriter.getvalue()}\nend\nshowpage\n', encoding='latin-1')
        if orientation is _Orientation.landscape:
            width, height = (height, width)
            bbox = (lly, llx, ury, urx)
        if is_eps or papertype == 'figure':
            paper_width, paper_height = orientation.swap_if_landscape(self.figure.get_size_inches())
        else:
            if papertype == 'auto':
                papertype = _get_papertype(width, height)
            paper_width, paper_height = papersize[papertype]
        psfrag_rotated = _convert_psfrags(tmppath, ps_renderer.psfrag, paper_width, paper_height, orientation.name)
        if mpl.rcParams['ps.usedistiller'] == 'ghostscript' or mpl.rcParams['text.usetex']:
            _try_distill(gs_distill, tmppath, is_eps, ptype=papertype, bbox=bbox, rotated=psfrag_rotated)
        elif mpl.rcParams['ps.usedistiller'] == 'xpdf':
            _try_distill(xpdf_distill, tmppath, is_eps, ptype=papertype, bbox=bbox, rotated=psfrag_rotated)
        _move_path_to_path_or_stream(tmppath, outfile)