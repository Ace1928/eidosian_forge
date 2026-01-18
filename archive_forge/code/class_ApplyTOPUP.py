import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class ApplyTOPUP(FSLCommand):
    """
    Interface for FSL topup, a tool for estimating and correcting
    susceptibility induced distortions.
    `General reference
    <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/ApplytopupUsersGuide>`_
    and `use example
    <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/ExampleTopupFollowedByApplytopup>`_.


    Examples
    --------

    >>> from nipype.interfaces.fsl import ApplyTOPUP
    >>> applytopup = ApplyTOPUP()
    >>> applytopup.inputs.in_files = ["epi.nii", "epi_rev.nii"]
    >>> applytopup.inputs.encoding_file = "topup_encoding.txt"
    >>> applytopup.inputs.in_topup_fieldcoef = "topup_fieldcoef.nii.gz"
    >>> applytopup.inputs.in_topup_movpar = "topup_movpar.txt"
    >>> applytopup.inputs.output_type = "NIFTI_GZ"
    >>> applytopup.cmdline # doctest: +ELLIPSIS
    'applytopup --datain=topup_encoding.txt --imain=epi.nii,epi_rev.nii --inindex=1,2 --topup=topup --out=epi_corrected.nii.gz'
    >>> res = applytopup.run() # doctest: +SKIP

    """
    _cmd = 'applytopup'
    input_spec = ApplyTOPUPInputSpec
    output_spec = ApplyTOPUPOutputSpec

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        if not isdefined(self.inputs.in_index):
            self.inputs.in_index = list(range(1, len(self.inputs.in_files) + 1))
        return super(ApplyTOPUP, self)._parse_inputs(skip=skip)

    def _format_arg(self, name, spec, value):
        if name == 'in_topup_fieldcoef':
            return spec.argstr % value.split('_fieldcoef')[0]
        return super(ApplyTOPUP, self)._format_arg(name, spec, value)