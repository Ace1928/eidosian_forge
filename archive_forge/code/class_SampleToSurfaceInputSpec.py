import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SampleToSurfaceInputSpec(FSTraitedSpec):
    source_file = File(exists=True, mandatory=True, argstr='--mov %s', desc='volume to sample values from')
    reference_file = File(exists=True, argstr='--ref %s', desc='reference volume (default is orig.mgz)')
    hemi = traits.Enum('lh', 'rh', mandatory=True, argstr='--hemi %s', desc='target hemisphere')
    surface = traits.String(argstr='--surf %s', desc='target surface (default is white)')
    reg_xors = ['reg_file', 'reg_header', 'mni152reg']
    reg_file = File(exists=True, argstr='--reg %s', mandatory=True, xor=reg_xors, desc='source-to-reference registration file')
    reg_header = traits.Bool(argstr='--regheader %s', requires=['subject_id'], mandatory=True, xor=reg_xors, desc='register based on header geometry')
    mni152reg = traits.Bool(argstr='--mni152reg', mandatory=True, xor=reg_xors, desc='source volume is in MNI152 space')
    apply_rot = traits.Tuple(traits.Float, traits.Float, traits.Float, argstr='--rot %.3f %.3f %.3f', desc='rotation angles (in degrees) to apply to reg matrix')
    apply_trans = traits.Tuple(traits.Float, traits.Float, traits.Float, argstr='--trans %.3f %.3f %.3f', desc='translation (in mm) to apply to reg matrix')
    override_reg_subj = traits.Bool(argstr='--srcsubject %s', requires=['subject_id'], desc='override the subject in the reg file header')
    sampling_method = traits.Enum('point', 'max', 'average', mandatory=True, argstr='%s', xor=['projection_stem'], requires=['sampling_range', 'sampling_units'], desc='how to sample -- at a point or at the max or average over a range')
    sampling_range = traits.Either(traits.Float, traits.Tuple(traits.Float, traits.Float, traits.Float), desc='sampling range - a point or a tuple of (min, max, step)')
    sampling_units = traits.Enum('mm', 'frac', desc="sampling range type -- either 'mm' or 'frac'")
    projection_stem = traits.String(mandatory=True, xor=['sampling_method'], desc='stem for precomputed linear estimates and volume fractions')
    smooth_vol = traits.Float(argstr='--fwhm %.3f', desc='smooth input volume (mm fwhm)')
    smooth_surf = traits.Float(argstr='--surf-fwhm %.3f', desc='smooth output surface (mm fwhm)')
    interp_method = traits.Enum('nearest', 'trilinear', argstr='--interp %s', desc='interpolation method')
    cortex_mask = traits.Bool(argstr='--cortex', xor=['mask_label'], desc='mask the target surface with hemi.cortex.label')
    mask_label = File(exists=True, argstr='--mask %s', xor=['cortex_mask'], desc='label file to mask output with')
    float2int_method = traits.Enum('round', 'tkregister', argstr='--float2int %s', desc='method to convert reg matrix values (default is round)')
    fix_tk_reg = traits.Bool(argstr='--fixtkreg', desc='make reg matrix round-compatible')
    subject_id = traits.String(desc='subject id')
    target_subject = traits.String(argstr='--trgsubject %s', desc='sample to surface of different subject than source')
    surf_reg = traits.Either(traits.Bool, traits.Str(), argstr='--surfreg %s', requires=['target_subject'], desc='use surface registration to target subject')
    ico_order = traits.Int(argstr='--icoorder %d', requires=['target_subject'], desc="icosahedron order when target_subject is 'ico'")
    reshape = traits.Bool(argstr='--reshape', xor=['no_reshape'], desc='reshape surface vector to fit in non-mgh format')
    no_reshape = traits.Bool(argstr='--noreshape', xor=['reshape'], desc='do not reshape surface vector (default)')
    reshape_slices = traits.Int(argstr='--rf %d', desc="number of 'slices' for reshaping")
    scale_input = traits.Float(argstr='--scale %.3f', desc='multiple all intensities by scale factor')
    frame = traits.Int(argstr='--frame %d', desc='save only one frame (0-based)')
    out_file = File(argstr='--o %s', genfile=True, desc='surface file to write')
    out_type = traits.Enum(filetypes + implicit_filetypes, argstr='--out_type %s', desc='output file type')
    hits_file = traits.Either(traits.Bool, File(exists=True), argstr='--srchit %s', desc='save image with number of hits at each voxel')
    hits_type = traits.Enum(filetypes, argstr='--srchit_type', desc='hits file type')
    vox_file = traits.Either(traits.Bool, File, argstr='--nvox %s', desc='text file with the number of voxels intersecting the surface')