import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def EscapeCppDefine(s):
    """Escapes a CPP define so that it will reach the compiler unaltered."""
    s = EscapeShellArgument(s)
    s = EscapeMakeVariableExpansion(s)
    return s.replace('#', '\\#')