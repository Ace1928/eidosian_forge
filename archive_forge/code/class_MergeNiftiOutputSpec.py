import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class MergeNiftiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Merged Nifti file')