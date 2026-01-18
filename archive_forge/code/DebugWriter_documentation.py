from __future__ import absolute_import
import os
import sys
import errno
from ..Compiler import Errors
from ..Compiler.StringEncoding import EncodedString

    Class to output debugging information for cygdb

    It writes debug information to cython_debug/cython_debug_info_<modulename>
    in the build directory.
    