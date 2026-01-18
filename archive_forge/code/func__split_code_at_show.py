import contextlib
import doctest
from io import StringIO
import itertools
import os
from os.path import relpath
from pathlib import Path
import re
import shutil
import sys
import textwrap
import traceback
from docutils.parsers.rst import directives, Directive
from docutils.parsers.rst.directives.images import Image
import jinja2  # Sphinx dependency.
from sphinx.errors import ExtensionError
import matplotlib
from matplotlib.backend_bases import FigureManagerBase
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers, cbook
def _split_code_at_show(text, function_name):
    """Split code at plt.show()."""
    is_doctest = contains_doctest(text)
    if function_name is None:
        parts = []
        part = []
        for line in text.split('\n'):
            if not is_doctest and line.startswith('plt.show(') or (is_doctest and line.strip() == '>>> plt.show()'):
                part.append(line)
                parts.append('\n'.join(part))
                part = []
            else:
                part.append(line)
        if '\n'.join(part).strip():
            parts.append('\n'.join(part))
    else:
        parts = [text]
    return (is_doctest, parts)