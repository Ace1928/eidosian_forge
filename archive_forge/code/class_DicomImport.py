import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class DicomImport(SPMCommand):
    """Uses spm to convert DICOM files to nii or img+hdr.

    Examples
    --------

    >>> import nipype.interfaces.spm.utils as spmu
    >>> di = spmu.DicomImport()
    >>> di.inputs.in_files = ['functional_1.dcm', 'functional_2.dcm']
    >>> di.run() # doctest: +SKIP
    """
    input_spec = DicomImportInputSpec
    output_spec = DicomImportOutputSpec
    _jobtype = 'util'
    _jobname = 'dicom'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == 'in_files':
            return np.array(val, dtype=object)
        if opt == 'output_dir':
            return np.array([val], dtype=object)
        if opt == 'output_dir':
            return os.path.abspath(val)
        if opt == 'icedims':
            if val:
                return 1
            return 0
        return super(DicomImport, self)._format_arg(opt, spec, val)

    def _run_interface(self, runtime):
        od = os.path.abspath(self.inputs.output_dir)
        if not os.path.isdir(od):
            os.mkdir(od)
        return super(DicomImport, self)._run_interface(runtime)

    def _list_outputs(self):
        from glob import glob
        outputs = self._outputs().get()
        od = os.path.abspath(self.inputs.output_dir)
        ext = self.inputs.format
        if self.inputs.output_dir_struct == 'flat':
            outputs['out_files'] = glob(os.path.join(od, '*.%s' % ext))
        elif self.inputs.output_dir_struct == 'series':
            outputs['out_files'] = glob(os.path.join(od, os.path.join('*', '*.%s' % ext)))
        elif self.inputs.output_dir_struct in ['patid', 'date_time', 'patname']:
            outputs['out_files'] = glob(os.path.join(od, os.path.join('*', '*', '*.%s' % ext)))
        elif self.inputs.output_dir_struct == 'patid_date':
            outputs['out_files'] = glob(os.path.join(od, os.path.join('*', '*', '*', '*.%s' % ext)))
        return outputs