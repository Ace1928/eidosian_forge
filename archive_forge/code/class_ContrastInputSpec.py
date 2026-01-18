import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ContrastInputSpec(FSTraitedSpec):
    subject_id = traits.String('subject_id', argstr='--s %s', usedefault=True, mandatory=True, desc='Subject being processed')
    hemisphere = traits.Enum('lh', 'rh', argstr='--%s-only', mandatory=True, desc='Hemisphere being processed')
    thickness = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/?h.thickness')
    white = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/<hemisphere>.white')
    annotation = File(mandatory=True, exists=True, desc='Input annotation file must be <subject_id>/label/<hemisphere>.aparc.annot')
    cortex = File(mandatory=True, exists=True, desc='Input cortex label must be <subject_id>/label/<hemisphere>.cortex.label')
    orig = File(exists=True, mandatory=True, desc='Implicit input file mri/orig.mgz')
    rawavg = File(exists=True, mandatory=True, desc='Implicit input file mri/rawavg.mgz')
    copy_inputs = traits.Bool(desc='If running as a node, set this to True.' + 'This will copy the input files to the node ' + 'directory.')