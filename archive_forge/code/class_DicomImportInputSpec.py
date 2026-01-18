import os
import numpy as np
from ...utils.filemanip import (
from ..base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from .base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
class DicomImportInputSpec(SPMCommandInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, field='data', desc='dicom files to be converted')
    output_dir_struct = traits.Enum('flat', 'series', 'patname', 'patid_date', 'patid', 'date_time', field='root', usedefault=True, desc='directory structure for the output.')
    output_dir = traits.Str('./converted_dicom', field='outdir', usedefault=True, desc='output directory.')
    format = traits.Enum('nii', 'img', field='convopts.format', usedefault=True, desc='output format.')
    icedims = traits.Bool(False, field='convopts.icedims', usedefault=True, desc='If image sorting fails, one can try using the additional SIEMENS ICEDims information to create unique filenames. Use this only if there would be multiple volumes with exactly the same file names.')