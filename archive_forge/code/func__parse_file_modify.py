from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _parse_file_modify(self, info):
    """Parse a filemodify command within a commit.

        :param info: a string in the format "mode dataref path"
          (where dataref might be the hard-coded literal 'inline').
        """
    params = info.split(b' ', 2)
    path = self._path(params[2])
    mode = self._mode(params[0])
    if params[1] == b'inline':
        dataref = None
        data = self._get_data(b'filemodify')
    else:
        dataref = params[1]
        data = None
    return commands.FileModifyCommand(path, mode, dataref, data)