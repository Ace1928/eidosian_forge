import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class DegreeCentralityInputSpec(CentralityInputSpec):
    """DegreeCentrality inputspec"""
    in_file = File(desc='input file to 3dDegreeCentrality', argstr='%s', position=-1, mandatory=True, exists=True, copyfile=False)
    sparsity = traits.Float(desc='only take the top percent of connections', argstr='-sparsity %f')
    oned_file = Str(desc='output filepath to text dump of correlation matrix', argstr='-out1D %s')