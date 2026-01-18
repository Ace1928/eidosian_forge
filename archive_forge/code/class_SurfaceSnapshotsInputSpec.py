import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SurfaceSnapshotsInputSpec(FSTraitedSpec):
    subject_id = traits.String(position=1, argstr='%s', mandatory=True, desc='subject to visualize')
    hemi = traits.Enum('lh', 'rh', position=2, argstr='%s', mandatory=True, desc='hemisphere to visualize')
    surface = traits.String(position=3, argstr='%s', mandatory=True, desc='surface to visualize')
    show_curv = traits.Bool(argstr='-curv', desc='show curvature', xor=['show_gray_curv'])
    show_gray_curv = traits.Bool(argstr='-gray', desc='show curvature in gray', xor=['show_curv'])
    overlay = File(exists=True, argstr='-overlay %s', desc='load an overlay volume/surface', requires=['overlay_range'])
    reg_xors = ['overlay_reg', 'identity_reg', 'mni152_reg']
    overlay_reg = File(exists=True, argstr='-overlay-reg %s', xor=reg_xors, desc='registration matrix file to register overlay to surface')
    identity_reg = traits.Bool(argstr='-overlay-reg-identity', xor=reg_xors, desc='use the identity matrix to register the overlay to the surface')
    mni152_reg = traits.Bool(argstr='-mni152reg', xor=reg_xors, desc='use to display a volume in MNI152 space on the average subject')
    overlay_range = traits.Either(traits.Float, traits.Tuple(traits.Float, traits.Float), traits.Tuple(traits.Float, traits.Float, traits.Float), desc='overlay range--either min, (min, max) or (min, mid, max)', argstr='%s')
    overlay_range_offset = traits.Float(argstr='-foffset %.3f', desc='overlay range will be symmetric around offset value')
    truncate_overlay = traits.Bool(argstr='-truncphaseflag 1', desc='truncate the overlay display')
    reverse_overlay = traits.Bool(argstr='-revphaseflag 1', desc='reverse the overlay display')
    invert_overlay = traits.Bool(argstr='-invphaseflag 1', desc='invert the overlay display')
    demean_overlay = traits.Bool(argstr='-zm', desc='remove mean from overlay')
    annot_file = File(exists=True, argstr='-annotation %s', xor=['annot_name'], desc='path to annotation file to display')
    annot_name = traits.String(argstr='-annotation %s', xor=['annot_file'], desc='name of annotation to display (must be in $subject/label directory')
    label_file = File(exists=True, argstr='-label %s', xor=['label_name'], desc='path to label file to display')
    label_name = traits.String(argstr='-label %s', xor=['label_file'], desc='name of label to display (must be in $subject/label directory')
    colortable = File(exists=True, argstr='-colortable %s', desc='load colortable file')
    label_under = traits.Bool(argstr='-labels-under', desc='draw label/annotation under overlay')
    label_outline = traits.Bool(argstr='-label-outline', desc='draw label/annotation as outline')
    patch_file = File(exists=True, argstr='-patch %s', desc='load a patch')
    orig_suffix = traits.String(argstr='-orig %s', desc='set the orig surface suffix string')
    sphere_suffix = traits.String(argstr='-sphere %s', desc='set the sphere.reg suffix string')
    show_color_scale = traits.Bool(argstr='-colscalebarflag 1', desc='display the color scale bar')
    show_color_text = traits.Bool(argstr='-colscaletext 1', desc='display text in the color scale bar')
    six_images = traits.Bool(desc='also take anterior and posterior snapshots')
    screenshot_stem = traits.String(desc='stem to use for screenshot file names')
    stem_template_args = traits.List(traits.String, requires=['screenshot_stem'], desc='input names to use as arguments for a string-formated stem template')
    tcl_script = File(exists=True, argstr='%s', genfile=True, desc='override default screenshot script')