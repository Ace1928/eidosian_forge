import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class GroupAndStackOutputSpec(TraitedSpec):
    out_list = traits.List(desc='List of output nifti files')