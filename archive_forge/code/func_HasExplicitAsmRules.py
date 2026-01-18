import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def HasExplicitAsmRules(self, spec):
    """Determine if there's an explicit rule for asm files. When there isn't we
        need to generate implicit rules to assemble .asm files."""
    return self._HasExplicitRuleForExtension(spec, 'asm')