import os
import re
from copy import deepcopy
import itertools as it
import glob
from glob import iglob
from ..utils.filemanip import split_filename
from .base import (
class Dcm2niiInputSpec(CommandLineInputSpec):
    source_names = InputMultiPath(File(exists=True), argstr='%s', position=-1, copyfile=False, mandatory=True, xor=['source_dir'])
    source_dir = Directory(exists=True, argstr='%s', position=-1, mandatory=True, xor=['source_names'])
    anonymize = traits.Bool(True, argstr='-a', usedefault=True, desc='Remove identifying information')
    config_file = File(exists=True, argstr='-b %s', genfile=True, desc='Load settings from specified inifile')
    collapse_folders = traits.Bool(True, argstr='-c', usedefault=True, desc='Collapse input folders')
    date_in_filename = traits.Bool(True, argstr='-d', usedefault=True, desc='Date in filename')
    events_in_filename = traits.Bool(True, argstr='-e', usedefault=True, desc='Events (series/acq) in filename')
    source_in_filename = traits.Bool(False, argstr='-f', usedefault=True, desc='Source filename')
    gzip_output = traits.Bool(False, argstr='-g', usedefault=True, desc='Gzip output (.gz)')
    id_in_filename = traits.Bool(False, argstr='-i', usedefault=True, desc='ID  in filename')
    nii_output = traits.Bool(True, argstr='-n', usedefault=True, desc='Save as .nii - if no, create .hdr/.img pair')
    output_dir = Directory(exists=True, argstr='-o %s', genfile=True, desc='Output dir - if unspecified, source directory is used')
    protocol_in_filename = traits.Bool(True, argstr='-p', usedefault=True, desc='Protocol in filename')
    reorient = traits.Bool(argstr='-r', desc='Reorient image to nearest orthogonal')
    spm_analyze = traits.Bool(argstr='-s', xor=['nii_output'], desc='SPM2/Analyze not SPM5/NIfTI')
    convert_all_pars = traits.Bool(True, argstr='-v', usedefault=True, desc='Convert every image in directory')
    reorient_and_crop = traits.Bool(False, argstr='-x', usedefault=True, desc='Reorient and crop 3D images')