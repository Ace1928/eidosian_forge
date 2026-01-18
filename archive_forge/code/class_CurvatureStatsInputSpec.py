import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class CurvatureStatsInputSpec(FSTraitedSpec):
    surface = File(argstr='-F %s', exists=True, desc='Specify surface file for CurvatureStats')
    curvfile1 = File(argstr='%s', position=-2, mandatory=True, exists=True, desc='Input file for CurvatureStats')
    curvfile2 = File(argstr='%s', position=-1, mandatory=True, exists=True, desc='Input file for CurvatureStats')
    hemisphere = traits.Enum('lh', 'rh', position=-3, argstr='%s', mandatory=True, desc='Hemisphere being processed')
    subject_id = traits.String('subject_id', usedefault=True, position=-4, argstr='%s', mandatory=True, desc='Subject being processed')
    out_file = File(argstr='-o %s', exists=False, name_source=['hemisphere'], name_template='%s.curv.stats', hash_files=False, desc='Output curvature stats file')
    min_max = traits.Bool(argstr='-m', desc='Output min / max information for the processed curvature.')
    values = traits.Bool(argstr='-G', desc='Triggers a series of derived curvature values')
    write = traits.Bool(argstr='--writeCurvatureFiles', desc='Write curvature files')
    copy_inputs = traits.Bool(desc='If running as a node, set this to True.' + 'This will copy the input files to the node ' + 'directory.')