import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def _WinIdlRule(self, source, prebuild, outputs):
    """Handle the implicit VS .idl rule for one source file. Fills |outputs|
        with files that are generated."""
    outdir, output, vars, flags = self.msvs_settings.GetIdlBuildData(source, self.config_name)
    outdir = self.GypPathToNinja(outdir)

    def fix_path(path, rel=None):
        path = os.path.join(outdir, path)
        dirname, basename = os.path.split(source)
        root, ext = os.path.splitext(basename)
        path = self.ExpandRuleVariables(path, root, dirname, source, ext, basename)
        if rel:
            path = os.path.relpath(path, rel)
        return path
    vars = [(name, fix_path(value, outdir)) for name, value in vars]
    output = [fix_path(p) for p in output]
    vars.append(('outdir', outdir))
    vars.append(('idlflags', flags))
    input = self.GypPathToNinja(source)
    self.ninja.build(output, 'idl', input, variables=vars, order_only=prebuild)
    outputs.extend(output)