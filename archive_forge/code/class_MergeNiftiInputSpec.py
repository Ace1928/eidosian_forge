import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class MergeNiftiInputSpec(NiftiGeneratorBaseInputSpec):
    in_files = traits.List(mandatory=True, desc='List of Nifti files to merge')
    sort_order = traits.Either(traits.Str(), traits.List(), desc='One or more meta data keys to sort files by.')
    merge_dim = traits.Int(desc='Dimension to merge along. If not specified, the last singular or non-existent dimension is used.')