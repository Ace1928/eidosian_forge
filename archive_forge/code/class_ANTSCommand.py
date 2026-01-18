import os
from packaging.version import Version, parse
from ... import logging
from ..base import CommandLine, CommandLineInputSpec, traits, isdefined, PackageInfo
class ANTSCommand(CommandLine):
    """Base class for ANTS interfaces"""
    input_spec = ANTSCommandInputSpec
    _num_threads = LOCAL_DEFAULT_NUMBER_OF_THREADS

    def __init__(self, **inputs):
        super(ANTSCommand, self).__init__(**inputs)
        self.inputs.on_trait_change(self._num_threads_update, 'num_threads')
        if not isdefined(self.inputs.num_threads):
            self.inputs.num_threads = self._num_threads
        else:
            self._num_threads_update()

    def _num_threads_update(self):
        self._num_threads = self.inputs.num_threads
        if self.inputs.num_threads == -1:
            if ALT_ITKv4_THREAD_LIMIT_VARIABLE in self.inputs.environ:
                del self.inputs.environ[ALT_ITKv4_THREAD_LIMIT_VARIABLE]
            if PREFERED_ITKv4_THREAD_LIMIT_VARIABLE in self.inputs.environ:
                del self.inputs.environ[PREFERED_ITKv4_THREAD_LIMIT_VARIABLE]
        else:
            self.inputs.environ.update({PREFERED_ITKv4_THREAD_LIMIT_VARIABLE: '%s' % self.inputs.num_threads})

    @staticmethod
    def _format_xarray(val):
        """Convenience method for converting input arrays [1,2,3] to
        commandline format '1x2x3'"""
        return 'x'.join([str(x) for x in val])

    @classmethod
    def set_default_num_threads(cls, num_threads):
        """Set the default number of threads for ITK calls

        This method is used to set the default number of ITK threads for all
        the ANTS interfaces. However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.num_threads
        """
        cls._num_threads = num_threads

    @property
    def version(self):
        return Info.version()