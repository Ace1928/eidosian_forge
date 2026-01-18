import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class DWIPreprocOutputSpec(TraitedSpec):
    out_file = File(argstr='%s', desc='output preprocessed image series')
    out_grad_mrtrix = File('grad.b', argstr='%s', usedefault=True, desc='preprocessed gradient file in mrtrix3 format')
    out_fsl_bvec = File('grad.bvecs', argstr='%s', usedefault=True, desc='exported fsl gradient bvec file')
    out_fsl_bval = File('grad.bvals', argstr='%s', usedefault=True, desc='exported fsl gradient bval file')