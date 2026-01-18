import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class NiftiGeneratorBaseInputSpec(TraitedSpec):
    out_format = traits.Str(desc='String which can be formatted with meta data to create the output filename(s)')
    out_ext = traits.Str('.nii.gz', usedefault=True, desc='Determines output file type')
    out_path = Directory(desc='output path, current working directory if not set')