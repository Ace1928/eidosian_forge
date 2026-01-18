import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegAverageInputSpec(NiftyRegCommandInputSpec):
    """Input Spec for RegAverage."""
    avg_files = traits.List(File(exist=True), position=1, argstr='-avg %s', sep=' ', xor=['avg_lts_files', 'avg_ref_file', 'demean1_ref_file', 'demean2_ref_file', 'demean3_ref_file', 'warp_files'], desc='Averaging of images/affine transformations')
    desc = 'Robust average of affine transformations'
    avg_lts_files = traits.List(File(exist=True), position=1, argstr='-avg_lts %s', sep=' ', xor=['avg_files', 'avg_ref_file', 'demean1_ref_file', 'demean2_ref_file', 'demean3_ref_file', 'warp_files'], desc=desc)
    desc = 'All input images are resampled into the space of <reference image> and averaged. A cubic spline interpolation scheme is used for resampling'
    avg_ref_file = File(position=1, argstr='-avg_tran %s', xor=['avg_files', 'avg_lts_files', 'demean1_ref_file', 'demean2_ref_file', 'demean3_ref_file'], requires=['warp_files'], desc=desc)
    desc = 'Average images and demean average image that have affine transformations to a common space'
    demean1_ref_file = File(position=1, argstr='-demean1 %s', xor=['avg_files', 'avg_lts_files', 'avg_ref_file', 'demean2_ref_file', 'demean3_ref_file'], requires=['warp_files'], desc=desc)
    desc = 'Average images and demean average image that have non-rigid transformations to a common space'
    demean2_ref_file = File(position=1, argstr='-demean2 %s', xor=['avg_files', 'avg_lts_files', 'avg_ref_file', 'demean1_ref_file', 'demean3_ref_file'], requires=['warp_files'], desc=desc)
    desc = 'Average images and demean average image that have linear and non-rigid transformations to a common space'
    demean3_ref_file = File(position=1, argstr='-demean3 %s', xor=['avg_files', 'avg_lts_files', 'avg_ref_file', 'demean1_ref_file', 'demean2_ref_file'], requires=['warp_files'], desc=desc)
    desc = 'transformation files and floating image pairs/triplets to the reference space'
    warp_files = traits.List(File(exist=True), position=-1, argstr='%s', sep=' ', xor=['avg_files', 'avg_lts_files'], desc=desc)
    out_file = File(genfile=True, position=0, desc='Output file name', argstr='%s')