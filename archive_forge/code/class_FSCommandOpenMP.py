import os
from ... import LooseVersion
from ...utils.filemanip import fname_presuffix
from ..base import (
class FSCommandOpenMP(FSCommand):
    """Support for FS commands that utilize OpenMP

    Sets the environment variable 'OMP_NUM_THREADS' to the number
    of threads specified by the input num_threads.
    """
    input_spec = FSTraitedSpecOpenMP
    _num_threads = None

    def __init__(self, **inputs):
        super(FSCommandOpenMP, self).__init__(**inputs)
        self.inputs.on_trait_change(self._num_threads_update, 'num_threads')
        if not self._num_threads:
            self._num_threads = os.environ.get('OMP_NUM_THREADS', None)
            if not self._num_threads:
                self._num_threads = os.environ.get('NSLOTS', None)
        if not isdefined(self.inputs.num_threads) and self._num_threads:
            self.inputs.num_threads = int(self._num_threads)
        self._num_threads_update()

    def _num_threads_update(self):
        if self.inputs.num_threads:
            self.inputs.environ.update({'OMP_NUM_THREADS': str(self.inputs.num_threads)})

    def run(self, **inputs):
        if 'num_threads' in inputs:
            self.inputs.num_threads = inputs['num_threads']
        self._num_threads_update()
        return super(FSCommandOpenMP, self).run(**inputs)