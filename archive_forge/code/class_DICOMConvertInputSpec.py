import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class DICOMConvertInputSpec(FSTraitedSpec):
    dicom_dir = Directory(exists=True, mandatory=True, desc='dicom directory from which to convert dicom files')
    base_output_dir = Directory(mandatory=True, desc='directory in which subject directories are created')
    subject_dir_template = traits.Str('S.%04d', usedefault=True, desc='template for subject directory name')
    subject_id = traits.Any(desc='subject identifier to insert into template')
    file_mapping = traits.List(traits.Tuple(traits.Str, traits.Str), desc='defines the output fields of interface')
    out_type = traits.Enum('niigz', MRIConvertInputSpec._filetypes, usedefault=True, desc='defines the type of output file produced')
    dicom_info = File(exists=True, desc='File containing summary information from mri_parse_sdcmdir')
    seq_list = traits.List(traits.Str, requires=['dicom_info'], desc='list of pulse sequence names to be converted.')
    ignore_single_slice = traits.Bool(requires=['dicom_info'], desc='ignore volumes containing a single slice')