import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TCorrMapInputSpec(AFNICommandInputSpec):
    in_file = File(exists=True, argstr='-input %s', mandatory=True, copyfile=False)
    seeds = File(exists=True, argstr='-seed %s', xor='seeds_width')
    mask = File(exists=True, argstr='-mask %s')
    automask = traits.Bool(argstr='-automask')
    polort = traits.Int(argstr='-polort %d')
    bandpass = traits.Tuple((traits.Float(), traits.Float()), argstr='-bpass %f %f')
    regress_out_timeseries = File(exists=True, argstr='-ort %s')
    blur_fwhm = traits.Float(argstr='-Gblur %f')
    seeds_width = traits.Float(argstr='-Mseed %f', xor='seeds')
    mean_file = File(argstr='-Mean %s', suffix='_mean', name_source='in_file')
    zmean = File(argstr='-Zmean %s', suffix='_zmean', name_source='in_file')
    qmean = File(argstr='-Qmean %s', suffix='_qmean', name_source='in_file')
    pmean = File(argstr='-Pmean %s', suffix='_pmean', name_source='in_file')
    _thresh_opts = ('absolute_threshold', 'var_absolute_threshold', 'var_absolute_threshold_normalize')
    thresholds = traits.List(traits.Int())
    absolute_threshold = File(argstr='-Thresh %f %s', suffix='_thresh', name_source='in_file', xor=_thresh_opts)
    var_absolute_threshold = File(argstr='-VarThresh %f %f %f %s', suffix='_varthresh', name_source='in_file', xor=_thresh_opts)
    var_absolute_threshold_normalize = File(argstr='-VarThreshN %f %f %f %s', suffix='_varthreshn', name_source='in_file', xor=_thresh_opts)
    correlation_maps = File(argstr='-CorrMap %s', name_source='in_file')
    correlation_maps_masked = File(argstr='-CorrMask %s', name_source='in_file')
    _expr_opts = ('average_expr', 'average_expr_nonzero', 'sum_expr')
    expr = Str()
    average_expr = File(argstr='-Aexpr %s %s', suffix='_aexpr', name_source='in_file', xor=_expr_opts)
    average_expr_nonzero = File(argstr='-Cexpr %s %s', suffix='_cexpr', name_source='in_file', xor=_expr_opts)
    sum_expr = File(argstr='-Sexpr %s %s', suffix='_sexpr', name_source='in_file', xor=_expr_opts)
    histogram_bin_numbers = traits.Int()
    histogram = File(name_source='in_file', argstr='-Hist %d %s', suffix='_hist')