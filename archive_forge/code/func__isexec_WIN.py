import io
import os
import pathlib
import re
import sys
from pprint import pformat
from IPython.core import magic_arguments
from IPython.core import oinspect
from IPython.core import page
from IPython.core.alias import AliasError, Alias
from IPython.core.error import UsageError
from IPython.core.magic import  (
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.openpy import source_to_unicode
from IPython.utils.process import abbrev_cwd
from IPython.utils.terminal import set_term_title
from traitlets import Bool
from warnings import warn
def _isexec_WIN(self, file):
    """
        Test for executable file on non POSIX system
        """
    return file.is_file() and self.execre.match(file.name) is not None