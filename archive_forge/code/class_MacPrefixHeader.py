import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
class MacPrefixHeader:
    """A class that helps with emulating Xcode's GCC_PREFIX_HEADER feature.

  This feature consists of several pieces:
  * If GCC_PREFIX_HEADER is present, all compilations in that project get an
    additional |-include path_to_prefix_header| cflag.
  * If GCC_PRECOMPILE_PREFIX_HEADER is present too, then the prefix header is
    instead compiled, and all other compilations in the project get an
    additional |-include path_to_compiled_header| instead.
    + Compiled prefix headers have the extension gch. There is one gch file for
      every language used in the project (c, cc, m, mm), since gch files for
      different languages aren't compatible.
    + gch files themselves are built with the target's normal cflags, but they
      obviously don't get the |-include| flag. Instead, they need a -x flag that
      describes their language.
    + All o files in the target need to depend on the gch file, to make sure
      it's built before any o file is built.

  This class helps with some of these tasks, but it needs help from the build
  system for writing dependencies to the gch files, for writing build commands
  for the gch files, and for figuring out the location of the gch files.
  """

    def __init__(self, xcode_settings, gyp_path_to_build_path, gyp_path_to_build_output):
        """If xcode_settings is None, all methods on this class are no-ops.

    Args:
        gyp_path_to_build_path: A function that takes a gyp-relative path,
            and returns a path relative to the build directory.
        gyp_path_to_build_output: A function that takes a gyp-relative path and
            a language code ('c', 'cc', 'm', or 'mm'), and that returns a path
            to where the output of precompiling that path for that language
            should be placed (without the trailing '.gch').
    """
        self.header = None
        self.compile_headers = False
        if xcode_settings:
            self.header = xcode_settings.GetPerTargetSetting('GCC_PREFIX_HEADER')
            self.compile_headers = xcode_settings.GetPerTargetSetting('GCC_PRECOMPILE_PREFIX_HEADER', default='NO') != 'NO'
        self.compiled_headers = {}
        if self.header:
            if self.compile_headers:
                for lang in ['c', 'cc', 'm', 'mm']:
                    self.compiled_headers[lang] = gyp_path_to_build_output(self.header, lang)
            self.header = gyp_path_to_build_path(self.header)

    def _CompiledHeader(self, lang, arch):
        assert self.compile_headers
        h = self.compiled_headers[lang]
        if arch:
            h += '.' + arch
        return h

    def GetInclude(self, lang, arch=None):
        """Gets the cflags to include the prefix header for language |lang|."""
        if self.compile_headers and lang in self.compiled_headers:
            return '-include %s' % self._CompiledHeader(lang, arch)
        elif self.header:
            return '-include %s' % self.header
        else:
            return ''

    def _Gch(self, lang, arch):
        """Returns the actual file name of the prefix header for language |lang|."""
        assert self.compile_headers
        return self._CompiledHeader(lang, arch) + '.gch'

    def GetObjDependencies(self, sources, objs, arch=None):
        """Given a list of source files and the corresponding object files, returns
    a list of (source, object, gch) tuples, where |gch| is the build-directory
    relative path to the gch file each object file depends on.  |compilable[i]|
    has to be the source file belonging to |objs[i]|."""
        if not self.header or not self.compile_headers:
            return []
        result = []
        for source, obj in zip(sources, objs):
            ext = os.path.splitext(source)[1]
            lang = {'.c': 'c', '.cpp': 'cc', '.cc': 'cc', '.cxx': 'cc', '.m': 'm', '.mm': 'mm'}.get(ext, None)
            if lang:
                result.append((source, obj, self._Gch(lang, arch)))
        return result

    def GetPchBuildCommands(self, arch=None):
        """Returns [(path_to_gch, language_flag, language, header)].
    |path_to_gch| and |header| are relative to the build directory.
    """
        if not self.header or not self.compile_headers:
            return []
        return [(self._Gch('c', arch), '-x c-header', 'c', self.header), (self._Gch('cc', arch), '-x c++-header', 'cc', self.header), (self._Gch('m', arch), '-x objective-c-header', 'm', self.header), (self._Gch('mm', arch), '-x objective-c++-header', 'mm', self.header)]