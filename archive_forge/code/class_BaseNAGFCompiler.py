import sys
import re
from numpy.distutils.fcompiler import FCompiler
class BaseNAGFCompiler(FCompiler):
    version_pattern = 'NAG.* Release (?P<version>[^(\\s]*)'

    def version_match(self, version_string):
        m = re.search(self.version_pattern, version_string)
        if m:
            return m.group('version')
        else:
            return None

    def get_flags_linker_so(self):
        return ['-Wl,-shared']

    def get_flags_opt(self):
        return ['-O4']

    def get_flags_arch(self):
        return []