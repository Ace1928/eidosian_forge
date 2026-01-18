import os
import sys
import re
import shlex
import itertools
from . import sysconfig
from ._modified import newer
from .ccompiler import CCompiler, gen_preprocess_options, gen_lib_options
from .errors import DistutilsExecError, CompileError, LibError, LinkError
from ._log import log
from ._macos_compat import compiler_fixup
@staticmethod
def _library_root(dir):
    """
        macOS users can specify an alternate SDK using'-isysroot'.
        Calculate the SDK root if it is specified.

        Note that, as of Xcode 7, Apple SDKs may contain textual stub
        libraries with .tbd extensions rather than the normal .dylib
        shared libraries installed in /.  The Apple compiler tool
        chain handles this transparently but it can cause problems
        for programs that are being built with an SDK and searching
        for specific libraries.  Callers of find_library_file need to
        keep in mind that the base filename of the returned SDK library
        file might have a different extension from that of the library
        file installed on the running system, for example:
          /Applications/Xcode.app/Contents/Developer/Platforms/
              MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/
              usr/lib/libedit.tbd
        vs
          /usr/lib/libedit.dylib
        """
    cflags = sysconfig.get_config_var('CFLAGS')
    match = re.search('-isysroot\\s*(\\S+)', cflags)
    apply_root = sys.platform == 'darwin' and match and (dir.startswith('/System/') or (dir.startswith('/usr/') and (not dir.startswith('/usr/local/'))))
    return os.path.join(match.group(1), dir[1:]) if apply_root else dir