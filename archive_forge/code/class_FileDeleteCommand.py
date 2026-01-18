from __future__ import division
import re
import stat
from .helpers import (
class FileDeleteCommand(FileCommand):

    def __init__(self, path):
        FileCommand.__init__(self, b'filedelete')
        self.path = check_path(path)

    def __bytes__(self):
        return b' '.join([b'D', format_path(self.path)])