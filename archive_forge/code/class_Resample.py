import os.path as op
import nibabel as nb
import numpy as np
from looseversion import LooseVersion
from ... import logging
from ..base import traits, TraitedSpec, File, isdefined
from .base import (
class Resample(DipyBaseInterface):
    """
    An interface to reslicing diffusion datasets.
    See
    http://nipy.org/dipy/examples_built/reslice_datasets.html#example-reslice-datasets.

    Example
    -------

    >>> import nipype.interfaces.dipy as dipy
    >>> reslice = dipy.Resample()
    >>> reslice.inputs.in_file = 'diffusion.nii'
    >>> reslice.run() # doctest: +SKIP
    """
    input_spec = ResampleInputSpec
    output_spec = ResampleOutputSpec

    def _run_interface(self, runtime):
        order = self.inputs.interp
        vox_size = None
        if isdefined(self.inputs.vox_size):
            vox_size = self.inputs.vox_size
        out_file = op.abspath(self._gen_outfilename())
        resample_proxy(self.inputs.in_file, order=order, new_zooms=vox_size, out_file=out_file)
        IFLOGGER.info('Resliced image saved as %s', out_file)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = op.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        fname, fext = op.splitext(op.basename(self.inputs.in_file))
        if fext == '.gz':
            fname, fext2 = op.splitext(fname)
            fext = fext2 + fext
        return op.abspath('%s_reslice%s' % (fname, fext))