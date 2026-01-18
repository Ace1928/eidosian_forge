import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def ExpandInputRoot(self, template, expansion, dirname):
    if '%(INPUT_ROOT)s' not in template and '%(INPUT_DIRNAME)s' not in template:
        return template
    path = template % {'INPUT_ROOT': expansion, 'INPUT_DIRNAME': dirname}
    return path