from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
class hooks(object):
    """
    Acts as a context manager. Saves the state of sys.modules and restores it
    after the 'with' block.

    Use like this:

    >>> from future import standard_library
    >>> with standard_library.hooks():
    ...     import http.client
    >>> import requests

    For this to work, http.client will be scrubbed from sys.modules after the
    'with' block. That way the modules imported in the 'with' block will
    continue to be accessible in the current namespace but not from any
    imported modules (like requests).
    """

    def __enter__(self):
        self.old_sys_modules = copy.copy(sys.modules)
        self.hooks_were_installed = detect_hooks()
        install_hooks()
        return self

    def __exit__(self, *args):
        if not self.hooks_were_installed:
            remove_hooks()