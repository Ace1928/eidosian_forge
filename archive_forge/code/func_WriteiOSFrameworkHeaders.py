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
def WriteiOSFrameworkHeaders(self, spec, outputs, prebuild):
    """Prebuild steps to generate hmap files and copy headers to destination."""
    framework = self.ComputeMacBundleOutput()
    all_sources = spec['sources']
    copy_headers = spec['mac_framework_headers']
    output = self.GypPathToUniqueOutput('headers.hmap')
    self.xcode_settings.header_map_path = output
    all_headers = map(self.GypPathToNinja, filter(lambda x: x.endswith('.h'), all_sources))
    variables = [('framework', framework), ('copy_headers', map(self.GypPathToNinja, copy_headers))]
    outputs.extend(self.ninja.build(output, 'compile_ios_framework_headers', all_headers, variables=variables, order_only=prebuild))