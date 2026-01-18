import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class FieldMap(SPMCommand):
    """Use the fieldmap toolbox from spm to calculate the voxel displacement map (VDM).

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=173

    .. important::

        This interface does not deal with real/imag magnitude images nor
        with the two phase files case.

    Examples
    --------
    >>> from nipype.interfaces.spm import FieldMap
    >>> fm = FieldMap()
    >>> fm.inputs.phase_file = 'phase.nii'
    >>> fm.inputs.magnitude_file = 'magnitude.nii'
    >>> fm.inputs.echo_times = (5.19, 7.65)
    >>> fm.inputs.blip_direction = 1
    >>> fm.inputs.total_readout_time = 15.6
    >>> fm.inputs.epi_file = 'epi.nii'
    >>> fm.run() # doctest: +SKIP

    """
    input_spec = FieldMapInputSpec
    output_spec = FieldMapOutputSpec
    _jobtype = 'tools'
    _jobname = 'fieldmap'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt in ['phase_file', 'magnitude_file', 'anat_file', 'epi_file']:
            return scans_for_fname(ensure_list(val))
        return super(FieldMap, self)._format_arg(opt, spec, val)

    def _parse_inputs(self):
        """validate spm fieldmap options if set to None ignore"""
        einputs = super(FieldMap, self)._parse_inputs()
        return [{'calculatevdm': einputs[0]}]

    def _list_outputs(self):
        outputs = self._outputs().get()
        jobtype = self.inputs.jobtype
        outputs['vdm'] = fname_presuffix(self.inputs.phase_file, prefix='vdm5_sc')
        return outputs