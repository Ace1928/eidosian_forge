import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class CopyGeom(FSLCommand):
    """Use fslcpgeom to copy the header geometry information to another image.
    Copy certain parts of the header information (image dimensions, voxel
    dimensions, voxel dimensions units string, image orientation/origin or
    qform/sform info) from one image to another. Note that only copies from
    Analyze to Analyze or Nifti to Nifti will work properly. Copying from
    different files will result in loss of information or potentially incorrect
    settings.
    """
    _cmd = 'fslcpgeom'
    input_spec = CopyGeomInputSpec
    output_spec = CopyGeomOutputSpec