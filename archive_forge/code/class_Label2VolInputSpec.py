import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class Label2VolInputSpec(FSTraitedSpec):
    label_file = InputMultiPath(File(exists=True), argstr='--label %s...', xor=('label_file', 'annot_file', 'seg_file', 'aparc_aseg'), copyfile=False, mandatory=True, desc='list of label files')
    annot_file = File(exists=True, argstr='--annot %s', xor=('label_file', 'annot_file', 'seg_file', 'aparc_aseg'), requires=('subject_id', 'hemi'), mandatory=True, copyfile=False, desc='surface annotation file')
    seg_file = File(exists=True, argstr='--seg %s', xor=('label_file', 'annot_file', 'seg_file', 'aparc_aseg'), mandatory=True, copyfile=False, desc='segmentation file')
    aparc_aseg = traits.Bool(argstr='--aparc+aseg', xor=('label_file', 'annot_file', 'seg_file', 'aparc_aseg'), mandatory=True, desc='use aparc+aseg.mgz in subjectdir as seg')
    template_file = File(exists=True, argstr='--temp %s', mandatory=True, desc='output template volume')
    reg_file = File(exists=True, argstr='--reg %s', xor=('reg_file', 'reg_header', 'identity'), desc='tkregister style matrix VolXYZ = R*LabelXYZ')
    reg_header = File(exists=True, argstr='--regheader %s', xor=('reg_file', 'reg_header', 'identity'), desc='label template volume')
    identity = traits.Bool(argstr='--identity', xor=('reg_file', 'reg_header', 'identity'), desc='set R=I')
    invert_mtx = traits.Bool(argstr='--invertmtx', desc='Invert the registration matrix')
    fill_thresh = traits.Range(0.0, 1.0, argstr='--fillthresh %g', desc='thresh : between 0 and 1')
    label_voxel_volume = traits.Float(argstr='--labvoxvol %f', desc='volume of each label point (def 1mm3)')
    proj = traits.Tuple(traits.Enum('abs', 'frac'), traits.Float, traits.Float, traits.Float, argstr='--proj %s %f %f %f', requires=('subject_id', 'hemi'), desc='project along surface normal')
    subject_id = traits.Str(argstr='--subject %s', desc='subject id')
    hemi = traits.Enum('lh', 'rh', argstr='--hemi %s', desc='hemisphere to use lh or rh')
    surface = traits.Str(argstr='--surf %s', desc='use surface instead of white')
    vol_label_file = File(argstr='--o %s', genfile=True, desc='output volume')
    label_hit_file = File(argstr='--hits %s', desc='file with each frame is nhits for a label')
    map_label_stat = File(argstr='--label-stat %s', desc='map the label stats field into the vol')
    native_vox2ras = traits.Bool(argstr='--native-vox2ras', desc='use native vox2ras xform instead of  tkregister-style')