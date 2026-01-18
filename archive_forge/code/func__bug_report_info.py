import argparse
import bdb
import locale
import multiprocessing
import os
import pdb
import sys
import traceback
from os import path
from typing import Any, List, Optional, TextIO
from docutils.utils import SystemMessage
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import Tee, format_exception_cut_frames, save_traceback
from sphinx.util.console import color_terminal, nocolor, red, terminal_safe  # type: ignore
from sphinx.util.docutils import docutils_namespace, patch_docutils
from sphinx.util.osutil import abspath, ensuredir
def _bug_report_info() -> int:
    from platform import platform, python_implementation
    import docutils
    import jinja2
    print('Please paste all output below into the bug report template\n\n')
    print('```text')
    print(f'Platform:              {sys.platform}; ({platform()})')
    print(f'Python version:        {sys.version})')
    print(f'Python implementation: {python_implementation()}')
    print(f'Sphinx version:        {sphinx.__display_version__}')
    print(f'Docutils version:      {docutils.__version__}')
    print(f'Jinja2 version:        {jinja2.__version__}')
    print('```')
    return 0