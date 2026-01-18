import filecmp
import gyp.common
import gyp.xcodeproj_file
import gyp.xcode_ninja
import errno
import os
import sys
import posixpath
import re
import shutil
import subprocess
import tempfile
def EscapeXcodeDefine(s):
    """We must escape the defines that we give to XCode so that it knows not to
     split on spaces and to respect backslash and quote literals. However, we
     must not quote the define, or Xcode will incorrectly interpret variables
     especially $(inherited)."""
    return re.sub(_xcode_define_re, '\\\\\\1', s)