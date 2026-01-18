import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
class ErrorMap(BaseInterface):
    """Calculates the error (distance) map between two input volumes.

    Example
    -------

    >>> errormap = ErrorMap()
    >>> errormap.inputs.in_ref = 'cont1.nii'
    >>> errormap.inputs.in_tst = 'cont2.nii'
    >>> res = errormap.run() # doctest: +SKIP
    """
    input_spec = ErrorMapInputSpec
    output_spec = ErrorMapOutputSpec
    _out_file = ''

    def _run_interface(self, runtime):
        nii_ref = nb.load(self.inputs.in_ref)
        ref_data = np.squeeze(nii_ref.dataobj)
        tst_data = np.squeeze(nb.load(self.inputs.in_tst).dataobj)
        assert ref_data.ndim == tst_data.ndim
        comps = 1
        mapshape = ref_data.shape
        if ref_data.ndim == 4:
            comps = ref_data.shape[-1]
            mapshape = ref_data.shape[:-1]
        if isdefined(self.inputs.mask):
            msk = np.asanyarray(nb.load(self.inputs.mask).dataobj)
            if mapshape != msk.shape:
                raise RuntimeError('Mask should match volume shape,                                    mask is %s and volumes are %s' % (list(msk.shape), list(mapshape)))
        else:
            msk = np.ones(shape=mapshape)
        mskvector = msk.reshape(-1)
        msk_idxs = np.where(mskvector == 1)
        refvector = ref_data.reshape(-1, comps)[msk_idxs].astype(np.float32)
        tstvector = tst_data.reshape(-1, comps)[msk_idxs].astype(np.float32)
        diffvector = refvector - tstvector
        if self.inputs.metric == 'sqeuclidean':
            errvector = diffvector ** 2
            if comps > 1:
                errvector = np.sum(errvector, axis=1)
            else:
                errvector = np.squeeze(errvector)
        elif self.inputs.metric == 'euclidean':
            errvector = np.linalg.norm(diffvector, axis=1)
        errvectorexp = np.zeros_like(mskvector, dtype=np.float32)
        errvectorexp[msk_idxs] = errvector
        self._distance = np.average(errvector)
        errmap = errvectorexp.reshape(mapshape)
        hdr = nii_ref.header.copy()
        hdr.set_data_dtype(np.float32)
        hdr['data_type'] = 16
        hdr.set_data_shape(mapshape)
        if not isdefined(self.inputs.out_map):
            fname, ext = op.splitext(op.basename(self.inputs.in_tst))
            if ext == '.gz':
                fname, ext2 = op.splitext(fname)
                ext = ext2 + ext
            self._out_file = op.abspath(fname + '_errmap' + ext)
        else:
            self._out_file = self.inputs.out_map
        nb.Nifti1Image(errmap.astype(np.float32), nii_ref.affine, hdr).to_filename(self._out_file)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_map'] = self._out_file
        outputs['distance'] = self._distance
        return outputs