import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class SplitNiftiInputSpec(NiftiGeneratorBaseInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Nifti file to split')
    split_dim = traits.Int(desc='Dimension to split along. If not specified, the last dimension is used.')