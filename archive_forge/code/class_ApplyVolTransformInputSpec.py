import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class ApplyVolTransformInputSpec(FSTraitedSpec):
    source_file = File(exists=True, argstr='--mov %s', copyfile=False, mandatory=True, desc='Input volume you wish to transform')
    transformed_file = File(desc='Output volume', argstr='--o %s', genfile=True)
    _targ_xor = ('target_file', 'tal', 'fs_target')
    target_file = File(exists=True, argstr='--targ %s', xor=_targ_xor, desc='Output template volume', mandatory=True)
    tal = traits.Bool(argstr='--tal', xor=_targ_xor, mandatory=True, desc='map to a sub FOV of MNI305 (with --reg only)')
    tal_resolution = traits.Float(argstr='--talres %.10f', desc='Resolution to sample when using tal')
    fs_target = traits.Bool(argstr='--fstarg', xor=_targ_xor, mandatory=True, requires=['reg_file'], desc='use orig.mgz from subject in regfile as target')
    _reg_xor = ('reg_file', 'lta_file', 'lta_inv_file', 'fsl_reg_file', 'xfm_reg_file', 'reg_header', 'mni_152_reg', 'subject')
    reg_file = File(exists=True, xor=_reg_xor, argstr='--reg %s', mandatory=True, desc='tkRAS-to-tkRAS matrix   (tkregister2 format)')
    lta_file = File(exists=True, xor=_reg_xor, argstr='--lta %s', mandatory=True, desc='Linear Transform Array file')
    lta_inv_file = File(exists=True, xor=_reg_xor, argstr='--lta-inv %s', mandatory=True, desc='LTA, invert')
    reg_file = File(exists=True, xor=_reg_xor, argstr='--reg %s', mandatory=True, desc='tkRAS-to-tkRAS matrix   (tkregister2 format)')
    fsl_reg_file = File(exists=True, xor=_reg_xor, argstr='--fsl %s', mandatory=True, desc='fslRAS-to-fslRAS matrix (FSL format)')
    xfm_reg_file = File(exists=True, xor=_reg_xor, argstr='--xfm %s', mandatory=True, desc='ScannerRAS-to-ScannerRAS matrix (MNI format)')
    reg_header = traits.Bool(xor=_reg_xor, argstr='--regheader', mandatory=True, desc='ScannerRAS-to-ScannerRAS matrix = identity')
    mni_152_reg = traits.Bool(xor=_reg_xor, argstr='--regheader', mandatory=True, desc='target MNI152 space')
    subject = traits.Str(xor=_reg_xor, argstr='--s %s', mandatory=True, desc='set matrix = identity and use subject for any templates')
    inverse = traits.Bool(desc='sample from target to source', argstr='--inv')
    interp = traits.Enum('trilin', 'nearest', 'cubic', argstr='--interp %s', desc='Interpolation method (<trilin> or nearest)')
    no_resample = traits.Bool(desc='Do not resample; just change vox2ras matrix', argstr='--no-resample')
    m3z_file = File(argstr='--m3z %s', desc='This is the morph to be applied to the volume. Unless the morph is in mri/transforms (eg.: for talairach.m3z computed by reconall), you will need to specify the full path to this morph and use the --noDefM3zPath flag.')
    no_ded_m3z_path = traits.Bool(argstr='--noDefM3zPath', requires=['m3z_file'], desc='To be used with the m3z flag. Instructs the code not to look for them3z morph in the default location (SUBJECTS_DIR/subj/mri/transforms), but instead just use the path indicated in --m3z.')
    invert_morph = traits.Bool(argstr='--inv-morph', requires=['m3z_file'], desc='Compute and use the inverse of the non-linear morph to resample the input volume. To be used by --m3z.')