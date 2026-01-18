import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TCorr1DInputSpec(AFNICommandInputSpec):
    xset = File(desc='3d+time dataset input', argstr=' %s', position=-2, mandatory=True, exists=True, copyfile=False)
    y_1d = File(desc='1D time series file input', argstr=' %s', position=-1, mandatory=True, exists=True)
    out_file = File(desc='output filename prefix', name_template='%s_correlation.nii.gz', argstr='-prefix %s', name_source='xset', keep_extension=True)
    pearson = traits.Bool(desc='Correlation is the normal Pearson correlation coefficient', argstr=' -pearson', xor=['spearman', 'quadrant', 'ktaub'], position=1)
    spearman = traits.Bool(desc='Correlation is the Spearman (rank) correlation coefficient', argstr=' -spearman', xor=['pearson', 'quadrant', 'ktaub'], position=1)
    quadrant = traits.Bool(desc='Correlation is the quadrant correlation coefficient', argstr=' -quadrant', xor=['pearson', 'spearman', 'ktaub'], position=1)
    ktaub = traits.Bool(desc="Correlation is the Kendall's tau_b correlation coefficient", argstr=' -ktaub', xor=['pearson', 'spearman', 'quadrant'], position=1)