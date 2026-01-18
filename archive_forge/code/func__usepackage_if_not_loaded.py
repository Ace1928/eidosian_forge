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
def _usepackage_if_not_loaded(package, *, option=None):
    """
    Output LaTeX code that loads a package (possibly with an option) if it
    hasn't been loaded yet.

    LaTeX cannot load twice a package with different options, so this helper
    can be used to protect against users loading arbitrary packages/options in
    their custom preamble.
    """
    option = f'[{option}]' if option is not None else ''
    return '\\makeatletter\\@ifpackageloaded{%(package)s}{}{\\usepackage%(option)s{%(package)s}}\\makeatother' % {'package': package, 'option': option}