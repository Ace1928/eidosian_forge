import os
import posixpath
import re
import stat
import threading
from mako import exceptions
from mako import util
from mako.template import Template
def _relativeize(self, filename):
    """Return the portion of a filename that is 'relative'
        to the directories in this lookup.

        """
    filename = posixpath.normpath(filename)
    for dir_ in self.directories:
        if filename[0:len(dir_)] == dir_:
            return filename[len(dir_):]
    else:
        return None