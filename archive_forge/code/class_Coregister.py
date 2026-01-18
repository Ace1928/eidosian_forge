import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class Coregister(SPMCommand):
    """Use spm_coreg for estimating cross-modality rigid body alignment

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=39

    Examples
    --------

    >>> import nipype.interfaces.spm as spm
    >>> coreg = spm.Coregister()
    >>> coreg.inputs.target = 'functional.nii'
    >>> coreg.inputs.source = 'structural.nii'
    >>> coreg.run() # doctest: +SKIP

    """
    input_spec = CoregisterInputSpec
    output_spec = CoregisterOutputSpec
    _jobtype = 'spatial'
    _jobname = 'coreg'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == 'target' or (opt == 'source' and self.inputs.jobtype != 'write'):
            return scans_for_fnames(ensure_list(val), keep4d=True)
        if opt == 'apply_to_files':
            return np.array(ensure_list(val), dtype=object)
        if opt == 'source' and self.inputs.jobtype == 'write':
            if isdefined(self.inputs.apply_to_files):
                return scans_for_fnames(val + self.inputs.apply_to_files)
            else:
                return scans_for_fnames(val)
        return super(Coregister, self)._format_arg(opt, spec, val)

    def _parse_inputs(self):
        """validate spm coregister options if set to None ignore"""
        if self.inputs.jobtype == 'write':
            einputs = super(Coregister, self)._parse_inputs(skip=('jobtype', 'apply_to_files'))
        else:
            einputs = super(Coregister, self)._parse_inputs(skip='jobtype')
        jobtype = self.inputs.jobtype
        return [{'%s' % jobtype: einputs[0]}]

    def _list_outputs(self):
        outputs = self._outputs().get()
        if self.inputs.jobtype == 'estimate':
            if isdefined(self.inputs.apply_to_files):
                outputs['coregistered_files'] = self.inputs.apply_to_files
            outputs['coregistered_source'] = self.inputs.source
        elif self.inputs.jobtype == 'write' or self.inputs.jobtype == 'estwrite':
            if isdefined(self.inputs.apply_to_files):
                outputs['coregistered_files'] = []
                for imgf in ensure_list(self.inputs.apply_to_files):
                    outputs['coregistered_files'].append(fname_presuffix(imgf, prefix=self.inputs.out_prefix))
            outputs['coregistered_source'] = []
            for imgf in ensure_list(self.inputs.source):
                outputs['coregistered_source'].append(fname_presuffix(imgf, prefix=self.inputs.out_prefix))
        return outputs