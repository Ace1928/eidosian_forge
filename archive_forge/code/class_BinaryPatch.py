import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
class BinaryPatch:

    def __init__(self, oldname, newname):
        self.oldname = oldname
        self.newname = newname

    def as_bytes(self):
        return b'Binary files %s and %s differ\n' % (self.oldname, self.newname)