import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ParcellationStatsInputSpec(FSTraitedSpec):
    subject_id = traits.String('subject_id', usedefault=True, position=-3, argstr='%s', mandatory=True, desc='Subject being processed')
    hemisphere = traits.Enum('lh', 'rh', position=-2, argstr='%s', mandatory=True, desc='Hemisphere being processed')
    wm = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/mri/wm.mgz')
    lh_white = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/lh.white')
    rh_white = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/rh.white')
    lh_pial = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/lh.pial')
    rh_pial = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/rh.pial')
    transform = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/mri/transforms/talairach.xfm')
    thickness = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/surf/?h.thickness')
    brainmask = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/mri/brainmask.mgz')
    aseg = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/mri/aseg.presurf.mgz')
    ribbon = File(mandatory=True, exists=True, desc='Input file must be <subject_id>/mri/ribbon.mgz')
    cortex_label = File(exists=True, desc='implicit input file {hemi}.cortex.label')
    surface = traits.String(position=-1, argstr='%s', desc="Input surface (e.g. 'white')")
    mgz = traits.Bool(argstr='-mgz', desc='Look for mgz files')
    in_cortex = File(argstr='-cortex %s', exists=True, desc='Input cortex label')
    in_annotation = File(argstr='-a %s', exists=True, xor=['in_label'], desc='compute properties for each label in the annotation file separately')
    in_label = File(argstr='-l %s', exists=True, xor=['in_annotatoin', 'out_color'], desc='limit calculations to specified label')
    tabular_output = traits.Bool(argstr='-b', desc='Tabular output')
    out_table = File(argstr='-f %s', exists=False, genfile=True, requires=['tabular_output'], desc='Table output to tablefile')
    out_color = File(argstr='-c %s', exists=False, genfile=True, xor=['in_label'], desc="Output annotation files's colortable to text file")
    copy_inputs = traits.Bool(desc='If running as a node, set this to True.' + 'This will copy the input files to the node ' + 'directory.')
    th3 = traits.Bool(argstr='-th3', requires=['cortex_label'], desc='turns on new vertex-wise volume calc for mris_anat_stats')