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
def WriteLink(self, spec, config_name, config, link_deps, compile_deps):
    """Write out a link step. Fills out target.binary. """
    if self.flavor != 'mac' or len(self.archs) == 1:
        return self.WriteLinkForArch(self.ninja, spec, config_name, config, link_deps, compile_deps)
    else:
        output = self.ComputeOutput(spec)
        inputs = [self.WriteLinkForArch(self.arch_subninjas[arch], spec, config_name, config, link_deps[arch], compile_deps, arch=arch) for arch in self.archs]
        extra_bindings = []
        build_output = output
        if not self.is_mac_bundle:
            self.AppendPostbuildVariable(extra_bindings, spec, output, output)
        if spec['type'] in ('shared_library', 'loadable_module') and (not self.is_mac_bundle):
            extra_bindings.append(('lib', output))
            self.ninja.build([output, output + '.TOC'], 'solipo', inputs, variables=extra_bindings)
        else:
            self.ninja.build(build_output, 'lipo', inputs, variables=extra_bindings)
        return output