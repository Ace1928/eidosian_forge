import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class ECMInputSpec(CentralityInputSpec):
    """ECM inputspec"""
    in_file = File(desc='input file to 3dECM', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    sparsity = traits.Float(desc='only take the top percent of connections', argstr='-sparsity %f')
    full = traits.Bool(desc='Full power method; enables thresholding; automatically selected if -thresh or -sparsity are set', argstr='-full')
    fecm = traits.Bool(desc='Fast centrality method; substantial speed increase but cannot accommodate thresholding; automatically selected if -thresh or -sparsity are not set', argstr='-fecm')
    shift = traits.Float(desc='shift correlation coefficients in similarity matrix to enforce non-negativity, s >= 0.0; default = 0.0 for -full, 1.0 for -fecm', argstr='-shift %f')
    scale = traits.Float(desc='scale correlation coefficients in similarity matrix to after shifting, x >= 0.0; default = 1.0 for -full, 0.5 for -fecm', argstr='-scale %f')
    eps = traits.Float(desc='sets the stopping criterion for the power iteration; :math:`l2\\|v_\\text{old} - v_\\text{new}\\| < eps\\|v_\\text{old}\\|`; default = 0.001', argstr='-eps %f')
    max_iter = traits.Int(desc='sets the maximum number of iterations to use in the power iteration; default = 1000', argstr='-max_iter %d')
    memory = traits.Float(desc='Limit memory consumption on system by setting the amount of GB to limit the algorithm to; default = 2GB', argstr='-memory %f')