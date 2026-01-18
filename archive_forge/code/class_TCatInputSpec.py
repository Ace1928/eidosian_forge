import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class TCatInputSpec(AFNICommandInputSpec):
    in_files = InputMultiPath(File(exists=True), desc='input file to 3dTcat', argstr=' %s', position=-1, mandatory=True, copyfile=False)
    out_file = File(name_template='%s_tcat', desc='output image file name', argstr='-prefix %s', name_source='in_files')
    rlt = traits.Enum('', '+', '++', argstr='-rlt%s', desc="Remove linear trends in each voxel time series loaded from each input dataset, SEPARATELY. Option -rlt removes the least squares fit of 'a+b*t' to each voxel time series. Option -rlt+ adds dataset mean back in. Option -rlt++ adds overall mean of all dataset timeseries back in.", position=1)
    verbose = traits.Bool(desc='Print out some verbose output as the program', argstr='-verb')