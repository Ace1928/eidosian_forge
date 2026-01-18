import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def XcodeArchsVariableMapping(archs, archs_including_64_bit=None):
    """Constructs a dictionary with expansion for $(ARCHS_STANDARD) variable,
  and optionally for $(ARCHS_STANDARD_INCLUDING_64_BIT)."""
    mapping = {'$(ARCHS_STANDARD)': archs}
    if archs_including_64_bit:
        mapping['$(ARCHS_STANDARD_INCLUDING_64_BIT)'] = archs_including_64_bit
    return mapping