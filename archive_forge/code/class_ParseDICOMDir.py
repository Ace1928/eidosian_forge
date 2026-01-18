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
class ParseDICOMDir(FSCommand):
    """Uses mri_parse_sdcmdir to get information from dicom directories

    Examples
    --------

    >>> from nipype.interfaces.freesurfer import ParseDICOMDir
    >>> dcminfo = ParseDICOMDir()
    >>> dcminfo.inputs.dicom_dir = '.'
    >>> dcminfo.inputs.sortbyrun = True
    >>> dcminfo.inputs.summarize = True
    >>> dcminfo.cmdline
    'mri_parse_sdcmdir --d . --o dicominfo.txt --sortbyrun --summarize'

    """
    _cmd = 'mri_parse_sdcmdir'
    input_spec = ParseDICOMDirInputSpec
    output_spec = ParseDICOMDirOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.dicom_info_file):
            outputs['dicom_info_file'] = os.path.join(os.getcwd(), self.inputs.dicom_info_file)
        return outputs