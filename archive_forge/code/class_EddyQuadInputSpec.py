import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EddyQuadInputSpec(FSLCommandInputSpec):
    base_name = traits.Str('eddy_corrected', usedefault=True, argstr='%s', desc='Basename (including path) for EDDY output files, i.e., corrected images and QC files', position=0)
    idx_file = File(exists=True, mandatory=True, argstr='--eddyIdx %s', desc='File containing indices for all volumes into acquisition parameters')
    param_file = File(exists=True, mandatory=True, argstr='--eddyParams %s', desc='File containing acquisition parameters')
    mask_file = File(exists=True, mandatory=True, argstr='--mask %s', desc='Binary mask file')
    bval_file = File(exists=True, mandatory=True, argstr='--bvals %s', desc='b-values file')
    bvec_file = File(exists=True, argstr='--bvecs %s', desc='b-vectors file - only used when <base_name>.eddy_residuals file is present')
    output_dir = traits.Str(name_template='%s.qc', name_source=['base_name'], argstr='--output-dir %s', desc="Output directory - default = '<base_name>.qc'")
    field = File(exists=True, argstr='--field %s', desc='TOPUP estimated field (in Hz)')
    slice_spec = File(exists=True, argstr='--slspec %s', desc='Text file specifying slice/group acquisition')
    verbose = traits.Bool(argstr='--verbose', desc='Display debug messages')