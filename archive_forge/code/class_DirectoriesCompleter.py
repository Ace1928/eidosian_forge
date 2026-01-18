from __future__ import absolute_import, division, print_function, unicode_literals
import os
import subprocess
from .compat import str, sys_encoding
class DirectoriesCompleter(_FilteredFilesCompleter):

    def __init__(self):
        _FilteredFilesCompleter.__init__(self, predicate=os.path.isdir)