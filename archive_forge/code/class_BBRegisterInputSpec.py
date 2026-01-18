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
class BBRegisterInputSpec(FSTraitedSpec):
    subject_id = traits.Str(argstr='--s %s', desc='freesurfer subject id', mandatory=True)
    source_file = File(argstr='--mov %s', desc='source file to be registered', mandatory=True, copyfile=False)
    init = traits.Enum('spm', 'fsl', 'header', argstr='--init-%s', mandatory=True, xor=['init_reg_file'], desc='initialize registration spm, fsl, header')
    init_reg_file = File(exists=True, argstr='--init-reg %s', desc='existing registration file', xor=['init'], mandatory=True)
    contrast_type = traits.Enum('t1', 't2', 'bold', 'dti', argstr='--%s', desc='contrast type of image', mandatory=True)
    intermediate_file = File(exists=True, argstr='--int %s', desc='Intermediate image, e.g. in case of partial FOV')
    reg_frame = traits.Int(argstr='--frame %d', xor=['reg_middle_frame'], desc='0-based frame index for 4D source file')
    reg_middle_frame = traits.Bool(argstr='--mid-frame', xor=['reg_frame'], desc='Register middle frame of 4D source file')
    out_reg_file = File(argstr='--reg %s', desc='output registration file', genfile=True)
    spm_nifti = traits.Bool(argstr='--spm-nii', desc='force use of nifti rather than analyze with SPM')
    epi_mask = traits.Bool(argstr='--epi-mask', desc='mask out B0 regions in stages 1 and 2')
    dof = traits.Enum(6, 9, 12, argstr='--%d', desc='number of transform degrees of freedom')
    fsldof = traits.Int(argstr='--fsl-dof %d', desc='degrees of freedom for initial registration (FSL)')
    out_fsl_file = traits.Either(traits.Bool, File, argstr='--fslmat %s', desc='write the transformation matrix in FSL FLIRT format')
    out_lta_file = traits.Either(traits.Bool, File, argstr='--lta %s', min_ver='5.2.0', desc='write the transformation matrix in LTA format')
    registered_file = traits.Either(traits.Bool, File, argstr='--o %s', desc='output warped sourcefile either True or filename')
    init_cost_file = traits.Either(traits.Bool, File, argstr='--initcost %s', desc='output initial registration cost file')