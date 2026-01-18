import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class SegStatsReconAllInputSpec(SegStatsInputSpec):
    subject_id = traits.String('subject_id', usedefault=True, argstr='--subject %s', mandatory=True, desc='Subject id being processed')
    ribbon = File(mandatory=True, exists=True, desc='Input file mri/ribbon.mgz')
    presurf_seg = File(exists=True, desc='Input segmentation volume')
    transform = File(mandatory=True, exists=True, desc='Input transform file')
    lh_orig_nofix = File(mandatory=True, exists=True, desc='Input lh.orig.nofix')
    rh_orig_nofix = File(mandatory=True, exists=True, desc='Input rh.orig.nofix')
    lh_white = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/lh.white')
    rh_white = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/rh.white')
    lh_pial = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/lh.pial')
    rh_pial = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/rh.pial')
    aseg = File(exists=True, desc='Mandatory implicit input in 5.3')
    copy_inputs = traits.Bool(desc='If running as a node, set this to True otherwise, this will copy the implicit inputs to the node directory.')