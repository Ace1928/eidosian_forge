import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class LookupMetaInputSpec(TraitedSpec):
    in_file = File(mandatory=True, exists=True, desc='The input Nifti file')
    meta_keys = traits.Either(traits.List(), traits.Dict(), mandatory=True, desc='List of meta data keys to lookup, or a dict where keys specify the meta data keys to lookup and the values specify the output names')