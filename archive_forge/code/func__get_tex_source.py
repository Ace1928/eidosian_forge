import functools
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, dviread
@classmethod
def _get_tex_source(cls, tex, fontsize):
    """Return the complete TeX source for processing a TeX string."""
    font_preamble, fontcmd = cls._get_font_preamble_and_command()
    baselineskip = 1.25 * fontsize
    return '\n'.join(['\\documentclass{article}', '% Pass-through \\mathdefault, which is used in non-usetex mode', '% to use the default text font but was historically suppressed', '% in usetex mode.', '\\newcommand{\\mathdefault}[1]{#1}', font_preamble, '\\usepackage[utf8]{inputenc}', '\\DeclareUnicodeCharacter{2212}{\\ensuremath{-}}', '% geometry is loaded before the custom preamble as ', '% convert_psfrags relies on a custom preamble to change the ', '% geometry.', '\\usepackage[papersize=72in, margin=1in]{geometry}', cls.get_custom_preamble(), '% Use `underscore` package to take care of underscores in text.', '% The [strings] option allows to use underscores in file names.', _usepackage_if_not_loaded('underscore', option='strings'), '% Custom packages (e.g. newtxtext) may already have loaded ', '% textcomp with different options.', _usepackage_if_not_loaded('textcomp'), '\\pagestyle{empty}', '\\begin{document}', '% The empty hbox ensures that a page is printed even for empty', '% inputs, except when using psfrag which gets confused by it.', '% matplotlibbaselinemarker is used by dviread to detect the', "% last line's baseline.", f'\\fontsize{{{fontsize}}}{{{baselineskip}}}%', '\\ifdefined\\psfrag\\else\\hbox{}\\fi%', f'{{{fontcmd} {tex}}}%', '\\end{document}'])