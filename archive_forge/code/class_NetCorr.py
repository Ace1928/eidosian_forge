import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class NetCorr(AFNICommand):
    """Calculate correlation matrix of a set of ROIs (using mean time series of
    each). Several networks may be analyzed simultaneously, one per brick.

    For complete details, see the `3dNetCorr Documentation
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dNetCorr.html>`_.

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> ncorr = afni.NetCorr()
    >>> ncorr.inputs.in_file = 'functional.nii'
    >>> ncorr.inputs.mask = 'mask.nii'
    >>> ncorr.inputs.in_rois = 'maps.nii'
    >>> ncorr.inputs.ts_wb_corr = True
    >>> ncorr.inputs.ts_wb_Z = True
    >>> ncorr.inputs.fish_z = True
    >>> ncorr.inputs.out_file = 'sub0.tp1.ncorr'
    >>> ncorr.cmdline
    '3dNetCorr -prefix sub0.tp1.ncorr -fish_z -inset functional.nii -in_rois maps.nii -mask mask.nii -ts_wb_Z -ts_wb_corr'
    >>> res = ncorr.run()  # doctest: +SKIP

    """
    _cmd = '3dNetCorr'
    input_spec = NetCorrInputSpec
    output_spec = NetCorrOutputSpec

    def _list_outputs(self):
        import glob
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            prefix = self._gen_fname(self.inputs.in_file, suffix='_netcorr')
        else:
            prefix = self.inputs.out_file
        odir = os.path.dirname(os.path.abspath(prefix))
        outputs['out_corr_matrix'] = glob.glob(os.path.join(odir, '*.netcc'))[0]
        if isdefined(self.inputs.ts_wb_corr) or isdefined(self.inputs.ts_Z_corr):
            corrdir = os.path.join(odir, prefix + '_000_INDIV')
            outputs['out_corr_maps'] = glob.glob(os.path.join(corrdir, '*.nii.gz'))
        return outputs