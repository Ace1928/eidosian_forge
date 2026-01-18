import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class CopyMetaInputSpec(TraitedSpec):
    src_file = File(mandatory=True, exists=True)
    dest_file = File(mandatory=True, exists=True)
    include_classes = traits.List(desc='List of specific meta data classifications to include. If not specified include everything.')
    exclude_classes = traits.List(desc='List of meta data classifications to exclude')