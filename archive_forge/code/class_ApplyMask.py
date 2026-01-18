import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ApplyMask(FSCommand):
    """Use Freesurfer's mri_mask to apply a mask to an image.

    The mask file need not be binarized; it can be thresholded above a given
    value before application. It can also optionally be transformed into input
    space with an LTA matrix.

    """
    _cmd = 'mri_mask'
    input_spec = ApplyMaskInputSpec
    output_spec = ApplyMaskOutputSpec