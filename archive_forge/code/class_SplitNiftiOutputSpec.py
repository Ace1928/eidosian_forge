import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class SplitNiftiOutputSpec(TraitedSpec):
    out_list = traits.List(File(exists=True), desc='Split Nifti files')