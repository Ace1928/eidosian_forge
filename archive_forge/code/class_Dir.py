import os
from fnmatch import fnmatch
from datetime import datetime
import operator
import re
class Dir(File):
    """
    Represents a directory created by a command.
    """
    file = False
    dir = True

    def __init__(self, base_path, path):
        self.base_path = base_path
        self.path = path
        self.full = os.path.join(base_path, path)
        self.size = 'N/A'
        self.mtime = 'N/A'

    def __repr__(self):
        return '<%s %s:%s>' % (self.__class__.__name__, self.base_path, self.path)

    def bytes__get(self):
        raise NotImplementedError("Directory %r doesn't have content" % self)
    bytes = property(bytes__get)