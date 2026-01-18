from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
def IsPathSafe(self):
    if self.is_remote:
        sep = '/'
    else:
        sep = os.sep * 2 if os.name == 'nt' else os.sep
    return not bool(re.search(Path._INVALID_PATH_FORMAT.format(sep=sep), self.path))