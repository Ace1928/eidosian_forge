from __future__ import division
import re
import stat
from .helpers import (
class FileRenameCommand(FileCommand):

    def __init__(self, old_path, new_path):
        FileCommand.__init__(self, b'filerename')
        self.old_path = check_path(old_path)
        self.new_path = check_path(new_path)

    def __bytes__(self):
        return b' '.join([b'R', format_path(self.old_path, quote_spaces=True), format_path(self.new_path)])