import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class MergeCSVFilesInputSpec(TraitedSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='Input comma-separated value (CSV) files')
    out_file = File('merged.csv', usedefault=True, desc='Output filename for merged CSV file')
    column_headings = traits.List(traits.Str, desc='List of column headings to save in merged CSV file        (must be equal to number of input files). If left undefined, these        will be pulled from the input filenames.')
    row_headings = traits.List(traits.Str, desc='List of row headings to save in merged CSV file        (must be equal to number of rows in the input files).')
    row_heading_title = traits.Str('label', usedefault=True, desc='Column heading for the row headings         added')
    extra_column_heading = traits.Str(desc='New heading to add for the added field.')
    extra_field = traits.Str(desc='New field to add to each row. This is useful for saving the        group or subject ID in the file.')