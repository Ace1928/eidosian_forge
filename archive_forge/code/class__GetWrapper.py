import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
class _GetWrapper:

    def __init__(self, parent, field, base_path, append=None):
        self.parent = parent
        self.field = field
        self.base_path = [base_path]
        self.append = append

    def __call__(self, name, map=None, prefix='', default=None):
        return self.parent._GetAndMunge(self.field, self.base_path + [name], default=default, prefix=prefix, append=self.append, map=map)