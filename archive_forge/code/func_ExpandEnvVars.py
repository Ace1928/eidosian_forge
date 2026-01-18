import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def ExpandEnvVars(string, expansions):
    """Expands ${VARIABLES}, $(VARIABLES), and $VARIABLES in string per the
  expansions list. If the variable expands to something that references
  another variable, this variable is expanded as well if it's in env --
  until no variables present in env are left."""
    for k, v in reversed(expansions):
        string = string.replace('${' + k + '}', v)
        string = string.replace('$(' + k + ')', v)
        string = string.replace('$' + k, v)
    return string