from __future__ import absolute_import, division, print_function, unicode_literals
import os
import subprocess
from .compat import str, sys_encoding
class SuppressCompleter(object):
    """
    A completer used to suppress the completion of specific arguments
    """

    def __init__(self):
        pass

    def suppress(self):
        """
        Decide if the completion should be suppressed
        """
        return True