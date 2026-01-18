import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FSLXCommand(FSLCommand):
    """
    Base support for ``xfibres`` and ``bedpostx``
    """
    input_spec = FSLXCommandInputSpec
    output_spec = FSLXCommandOutputSpec

    def _run_interface(self, runtime):
        self._out_dir = os.getcwd()
        runtime = super(FSLXCommand, self)._run_interface(runtime)
        if runtime.stderr:
            self.raise_exception(runtime)
        return runtime

    def _list_outputs(self, out_dir=None):
        outputs = self.output_spec().get()
        n_fibres = self.inputs.n_fibres
        if not out_dir:
            if isdefined(self.inputs.logdir):
                out_dir = os.path.abspath(self.inputs.logdir)
            else:
                out_dir = os.path.abspath('logdir')
        multi_out = ['dyads', 'fsamples', 'mean_fsamples', 'phsamples', 'thsamples']
        single_out = ['mean_dsamples', 'mean_S0samples']
        for k in single_out:
            outputs[k] = self._gen_fname(k, cwd=out_dir)
        if isdefined(self.inputs.rician) and self.inputs.rician:
            outputs['mean_tausamples'] = self._gen_fname('mean_tausamples', cwd=out_dir)
        for k in multi_out:
            outputs[k] = []
        for i in range(1, n_fibres + 1):
            outputs['fsamples'].append(self._gen_fname('f%dsamples' % i, cwd=out_dir))
            outputs['mean_fsamples'].append(self._gen_fname('mean_f%dsamples' % i, cwd=out_dir))
        for i in range(1, n_fibres + 1):
            outputs['dyads'].append(self._gen_fname('dyads%d' % i, cwd=out_dir))
            outputs['phsamples'].append(self._gen_fname('ph%dsamples' % i, cwd=out_dir))
            outputs['thsamples'].append(self._gen_fname('th%dsamples' % i, cwd=out_dir))
        return outputs