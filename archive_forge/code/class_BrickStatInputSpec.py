import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class BrickStatInputSpec(CommandLineInputSpec):
    in_file = File(desc='input file to 3dmaskave', argstr='%s', position=-1, mandatory=True, exists=True)
    mask = File(desc='-mask dset = use dset as mask to include/exclude voxels', argstr='-mask %s', position=2, exists=True)
    min = traits.Bool(desc='print the minimum value in dataset', argstr='-min', position=1)
    slow = traits.Bool(desc='read the whole dataset to find the min and max values', argstr='-slow')
    max = traits.Bool(desc='print the maximum value in the dataset', argstr='-max')
    mean = traits.Bool(desc='print the mean value in the dataset', argstr='-mean')
    sum = traits.Bool(desc='print the sum of values in the dataset', argstr='-sum')
    var = traits.Bool(desc='print the variance in the dataset', argstr='-var')
    percentile = traits.Tuple(traits.Float, traits.Float, traits.Float, desc='p0 ps p1 write the percentile values starting at p0% and ending at p1% at a step of ps%. only one sub-brick is accepted.', argstr='-percentile %.3f %.3f %.3f')