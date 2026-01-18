import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class OutlierCountInputSpec(CommandLineInputSpec):
    in_file = File(argstr='%s', mandatory=True, exists=True, position=-2, desc='input dataset')
    mask = File(exists=True, argstr='-mask %s', xor=['autoclip', 'automask'], desc='only count voxels within the given mask')
    qthr = traits.Range(value=0.001, low=0.0, high=1.0, usedefault=True, argstr='-qthr %.5f', desc='indicate a value for q to compute alpha')
    autoclip = traits.Bool(False, usedefault=True, argstr='-autoclip', xor=['mask'], desc='clip off small voxels')
    automask = traits.Bool(False, usedefault=True, argstr='-automask', xor=['mask'], desc='clip off small voxels')
    fraction = traits.Bool(False, usedefault=True, argstr='-fraction', desc='write out the fraction of masked voxels which are outliers at each timepoint')
    interval = traits.Bool(False, usedefault=True, argstr='-range', desc='write out the median + 3.5 MAD of outlier count with each timepoint')
    save_outliers = traits.Bool(False, usedefault=True, desc='enables out_file option')
    outliers_file = File(name_template='%s_outliers', argstr='-save %s', name_source=['in_file'], output_name='out_outliers', keep_extension=True, desc='output image file name')
    polort = traits.Int(argstr='-polort %d', desc='detrend each voxel timeseries with polynomials')
    legendre = traits.Bool(False, usedefault=True, argstr='-legendre', desc='use Legendre polynomials')
    out_file = File(name_template='%s_outliers', name_source=['in_file'], keep_extension=False, desc='capture standard output')