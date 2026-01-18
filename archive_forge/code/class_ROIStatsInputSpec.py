import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class ROIStatsInputSpec(CommandLineInputSpec):
    in_file = File(desc='input dataset', argstr='%s', position=-2, mandatory=True, exists=True)
    mask = File(desc='input mask', argstr='-mask %s', position=3, exists=True, deprecated='1.1.4', new_name='mask_file')
    mask_file = File(desc='input mask', argstr='-mask %s', exists=True)
    mask_f2short = traits.Bool(desc='Tells the program to convert a float mask to short integers, by simple rounding.', argstr='-mask_f2short')
    num_roi = traits.Int(desc="Forces the assumption that the mask dataset's ROIs are denoted by 1 to n inclusive.  Normally, the program figures out the ROIs on its own.  This option is useful if a) you are certain that the mask dataset has no values outside the range [0 n], b) there may be some ROIs missing between [1 n] in the mask data-set and c) you want those columns in the output any-way so the output lines up with the output from other invocations of 3dROIstats.", argstr='-numroi %s')
    zerofill = traits.Str(requires=['num_roi'], desc="For ROI labels not found, use the provided string instead of a '0' in the output file. Only active if `num_roi` is enabled.", argstr='-zerofill %s')
    roisel = File(exists=True, desc="Only considers ROIs denoted by values found in the specified file. Note that the order of the ROIs as specified in the file is not preserved. So an SEL.1D of '2 8 20' produces the same output as '8 20 2'", argstr='-roisel %s')
    debug = traits.Bool(desc='print debug information', argstr='-debug')
    quiet = traits.Bool(desc='execute quietly', argstr='-quiet')
    nomeanout = traits.Bool(desc='Do not include the (zero-inclusive) mean among computed stats', argstr='-nomeanout')
    nobriklab = traits.Bool(desc='Do not print the sub-brick label next to its index', argstr='-nobriklab')
    format1D = traits.Bool(xor=['format1DR'], desc='Output results in a 1D format that includes commented labels', argstr='-1Dformat')
    format1DR = traits.Bool(xor=['format1D'], desc='Output results in a 1D format that includes uncommented labels. May not work optimally with typical 1D functions, but is useful for R functions.', argstr='-1DRformat')
    _stat_names = ['mean', 'sum', 'voxels', 'minmax', 'sigma', 'median', 'mode', 'summary', 'zerominmax', 'zerosigma', 'zeromedian', 'zeromode']
    stat = InputMultiObject(traits.Enum(_stat_names), desc='Statistics to compute. Options include:\n\n * mean       =   Compute the mean using only non_zero voxels.\n                  Implies the opposite for the mean computed\n                  by default.\n * median     =   Compute the median of nonzero voxels\n * mode       =   Compute the mode of nonzero voxels.\n                  (integral valued sets only)\n * minmax     =   Compute the min/max of nonzero voxels\n * sum        =   Compute the sum using only nonzero voxels.\n * voxels     =   Compute the number of nonzero voxels\n * sigma      =   Compute the standard deviation of nonzero\n                  voxels\n\nStatistics that include zero-valued voxels:\n\n * zerominmax =   Compute the min/max of all voxels.\n * zerosigma  =   Compute the standard deviation of all\n                  voxels.\n * zeromedian =   Compute the median of all voxels.\n * zeromode   =   Compute the mode of all voxels.\n * summary    =   Only output a summary line with the grand\n                  mean across all briks in the input dataset.\n                  This option cannot be used with nomeanout.\n\nMore that one option can be specified.', argstr='%s...')
    out_file = File(name_template='%s_roistat.1D', desc='output file', keep_extension=False, argstr='> %s', name_source='in_file', position=-1)