import os
import sys
import warnings
from io import StringIO
from typing import Any, Dict, Optional
from sphinx.application import Sphinx
from sphinx.cmd.build import handle_exception
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.util.console import color_terminal, nocolor
from sphinx.util.docutils import docutils_namespace, patch_docutils
from sphinx.util.osutil import abspath
def _guess_source_dir(self) -> str:
    for guess in ('doc', 'docs'):
        if not os.path.isdir(guess):
            continue
        for root, _dirnames, filenames in os.walk(guess):
            if 'conf.py' in filenames:
                return root
    return os.curdir