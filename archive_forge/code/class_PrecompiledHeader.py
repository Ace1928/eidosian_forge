import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
class PrecompiledHeader:
    """Helper to generate dependencies and build rules to handle generation of
    precompiled headers. Interface matches the GCH handler in xcode_emulation.py.
    """

    def __init__(self, settings, config, gyp_to_build_path, gyp_to_unique_output, obj_ext):
        self.settings = settings
        self.config = config
        pch_source = self.settings.msvs_precompiled_source[self.config]
        self.pch_source = gyp_to_build_path(pch_source)
        filename, _ = os.path.splitext(pch_source)
        self.output_obj = gyp_to_unique_output(filename + obj_ext).lower()

    def _PchHeader(self):
        """Get the header that will appear in an #include line for all source
        files."""
        return self.settings.msvs_precompiled_header[self.config]

    def GetObjDependencies(self, sources, objs, arch):
        """Given a list of sources files and the corresponding object files,
        returns a list of the pch files that should be depended upon. The
        additional wrapping in the return value is for interface compatibility
        with make.py on Mac, and xcode_emulation.py."""
        assert arch is None
        if not self._PchHeader():
            return []
        pch_ext = os.path.splitext(self.pch_source)[1]
        for source in sources:
            if _LanguageMatchesForPch(os.path.splitext(source)[1], pch_ext):
                return [(None, None, self.output_obj)]
        return []

    def GetPchBuildCommands(self, arch):
        """Not used on Windows as there are no additional build steps required
        (instead, existing steps are modified in GetFlagsModifications below)."""
        return []

    def GetFlagsModifications(self, input, output, implicit, command, cflags_c, cflags_cc, expand_special):
        """Get the modified cflags and implicit dependencies that should be used
        for the pch compilation step."""
        if input == self.pch_source:
            pch_output = ['/Yc' + self._PchHeader()]
            if command == 'cxx':
                return ([('cflags_cc', map(expand_special, cflags_cc + pch_output))], self.output_obj, [])
            elif command == 'cc':
                return ([('cflags_c', map(expand_special, cflags_c + pch_output))], self.output_obj, [])
        return ([], output, implicit)