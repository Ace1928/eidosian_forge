import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class WarpUtils(FSLCommand):
    """Use FSL `fnirtfileutils <http://fsl.fmrib.ox.ac.uk/fsl/fsl-4.1.9/fnirt/warp_utils.html>`_
    to convert field->coefficients, coefficients->field, coefficients->other_coefficients etc


    Examples
    --------

    >>> from nipype.interfaces.fsl import WarpUtils
    >>> warputils = WarpUtils()
    >>> warputils.inputs.in_file = "warpfield.nii"
    >>> warputils.inputs.reference = "T1.nii"
    >>> warputils.inputs.out_format = 'spline'
    >>> warputils.inputs.warp_resolution = (10,10,10)
    >>> warputils.inputs.output_type = "NIFTI_GZ"
    >>> warputils.cmdline # doctest: +ELLIPSIS
    'fnirtfileutils --in=warpfield.nii --outformat=spline --ref=T1.nii --warpres=10.0000,10.0000,10.0000 --out=warpfield_coeffs.nii.gz'
    >>> res = invwarp.run() # doctest: +SKIP


    """
    input_spec = WarpUtilsInputSpec
    output_spec = WarpUtilsOutputSpec
    _cmd = 'fnirtfileutils'

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        suffix = 'field'
        if isdefined(self.inputs.out_format) and self.inputs.out_format == 'spline':
            suffix = 'coeffs'
        trait_spec = self.inputs.trait('out_file')
        trait_spec.name_template = '%s_' + suffix
        if self.inputs.write_jacobian:
            if not isdefined(self.inputs.out_jacobian):
                jac_spec = self.inputs.trait('out_jacobian')
                jac_spec.name_source = ['in_file']
                jac_spec.name_template = '%s_jac'
                jac_spec.output_name = 'out_jacobian'
        else:
            skip += ['out_jacobian']
        skip += ['write_jacobian']
        return super(WarpUtils, self)._parse_inputs(skip=skip)