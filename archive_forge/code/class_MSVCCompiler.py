import os
from distutils.msvccompiler import MSVCCompiler as _MSVCCompiler
from .system_info import platform_bits
class MSVCCompiler(_MSVCCompiler):

    def __init__(self, verbose=0, dry_run=0, force=0):
        _MSVCCompiler.__init__(self, verbose, dry_run, force)

    def initialize(self):
        environ_lib = os.getenv('lib', '')
        environ_include = os.getenv('include', '')
        _MSVCCompiler.initialize(self)
        os.environ['lib'] = _merge(environ_lib, os.environ['lib'])
        os.environ['include'] = _merge(environ_include, os.environ['include'])
        if platform_bits == 32:
            self.compile_options += ['/arch:SSE2']
            self.compile_options_debug += ['/arch:SSE2']