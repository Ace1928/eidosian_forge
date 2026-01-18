import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class TCatSBInputSpec(AFNICommandInputSpec):
    in_files = traits.List(traits.Tuple(File(exists=True), Str()), desc="List of tuples of file names and subbrick selectors as strings.Don't forget to protect the single quotes in the subbrick selectorso the contents are protected from the command line interpreter.", argstr='%s%s ...', position=-1, mandatory=True, copyfile=False)
    out_file = File(desc='output image file name', argstr='-prefix %s', genfile=True)
    rlt = traits.Enum('', '+', '++', argstr='-rlt%s', desc="Remove linear trends in each voxel time series loaded from each input dataset, SEPARATELY. Option -rlt removes the least squares fit of 'a+b*t' to each voxel time series. Option -rlt+ adds dataset mean back in. Option -rlt++ adds overall mean of all dataset timeseries back in.", position=1)