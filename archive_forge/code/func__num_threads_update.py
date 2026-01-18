import os
from packaging.version import Version, parse
from ... import logging
from ..base import CommandLine, CommandLineInputSpec, traits, isdefined, PackageInfo
def _num_threads_update(self):
    self._num_threads = self.inputs.num_threads
    if self.inputs.num_threads == -1:
        if ALT_ITKv4_THREAD_LIMIT_VARIABLE in self.inputs.environ:
            del self.inputs.environ[ALT_ITKv4_THREAD_LIMIT_VARIABLE]
        if PREFERED_ITKv4_THREAD_LIMIT_VARIABLE in self.inputs.environ:
            del self.inputs.environ[PREFERED_ITKv4_THREAD_LIMIT_VARIABLE]
    else:
        self.inputs.environ.update({PREFERED_ITKv4_THREAD_LIMIT_VARIABLE: '%s' % self.inputs.num_threads})