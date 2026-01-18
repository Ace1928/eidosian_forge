import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _LanguageMatchesForPch(source_ext, pch_source_ext):
    c_exts = ('.c',)
    cc_exts = ('.cc', '.cxx', '.cpp')
    return source_ext in c_exts and pch_source_ext in c_exts or (source_ext in cc_exts and pch_source_ext in cc_exts)