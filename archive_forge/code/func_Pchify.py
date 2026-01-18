import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def Pchify(self, path, lang):
    """Convert a prefix header path to its output directory form."""
    path = self.Absolutify(path)
    if '$(' in path:
        path = path.replace('$(obj)/', f'$(obj).{self.toolset}/$(TARGET)/pch-{lang}')
        return path
    return f'$(obj).{self.toolset}/$(TARGET)/pch-{lang}/{path}'