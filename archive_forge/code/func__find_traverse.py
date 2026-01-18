import os
from fnmatch import fnmatch
from datetime import datetime
import operator
import re
def _find_traverse(self, path, result):
    full = os.path.join(self.base_path, path)
    if os.path.isdir(full):
        if path:
            result[path] = Dir(self.base_path, path)
        for fn in os.listdir(full):
            fn = os.path.join(path, fn)
            if self._ignore_file(fn):
                continue
            self._find_traverse(fn, result)
    else:
        result[path] = File(self.base_path, path)