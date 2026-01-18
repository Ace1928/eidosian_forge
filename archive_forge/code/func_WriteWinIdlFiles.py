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
def WriteWinIdlFiles(self, spec, prebuild):
    """Writes rules to match MSVS's implicit idl handling."""
    assert self.flavor == 'win'
    if self.msvs_settings.HasExplicitIdlRulesOrActions(spec):
        return []
    outputs = []
    for source in filter(lambda x: x.endswith('.idl'), spec['sources']):
        self._WinIdlRule(source, prebuild, outputs)
    return outputs