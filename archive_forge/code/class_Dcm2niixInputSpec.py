import os
import re
from copy import deepcopy
import itertools as it
import glob
from glob import iglob
from ..utils.filemanip import split_filename
from .base import (
class Dcm2niixInputSpec(CommandLineInputSpec):
    source_names = InputMultiPath(File(exists=True), argstr='%s', position=-1, copyfile=False, mandatory=True, desc='A set of filenames to be converted. Note that the current version (1.0.20180328) of dcm2niix converts any files in the directory. To only convert specific files they should be in an isolated directory', xor=['source_dir'])
    source_dir = Directory(exists=True, argstr='%s', position=-1, mandatory=True, desc='A directory containing dicom files to be converted', xor=['source_names'])
    out_filename = traits.Str(argstr='-f %s', desc='Output filename template (%a=antenna (coil) number, %c=comments, %d=description, %e=echo number, %f=folder name, %i=ID of patient, %j=seriesInstanceUID, %k=studyInstanceUID, %m=manufacturer, %n=name of patient, %p=protocol, %s=series number, %t=time, %u=acquisition number, %v=vendor, %x=study ID; %z=sequence name)')
    output_dir = Directory('.', usedefault=True, exists=True, argstr='-o %s', desc='Output directory')
    bids_format = traits.Bool(True, argstr='-b', usedefault=True, desc='Create a BIDS sidecar file')
    anon_bids = traits.Bool(argstr='-ba', requires=['bids_format'], desc='Anonymize BIDS')
    compress = traits.Enum('y', 'i', 'n', '3', argstr='-z %s', usedefault=True, desc='Gzip compress images - [y=pigz, i=internal, n=no, 3=no,3D]')
    merge_imgs = traits.Bool(False, argstr='-m', usedefault=True, desc='merge 2D slices from same series')
    single_file = traits.Bool(False, argstr='-s', usedefault=True, desc='Single file mode')
    verbose = traits.Bool(False, argstr='-v', usedefault=True, desc='Verbose output')
    crop = traits.Bool(False, argstr='-x', usedefault=True, desc='Crop 3D T1 acquisitions')
    has_private = traits.Bool(False, argstr='-t', usedefault=True, desc='Text notes including private patient details')
    compression = traits.Enum(1, 2, 3, 4, 5, 6, 7, 8, 9, argstr='-%d', desc='Gz compression level (1=fastest, 9=smallest)')
    comment = traits.Str(argstr='-c %s', desc='Comment stored as NIfTI aux_file')
    ignore_deriv = traits.Bool(argstr='-i', desc='Ignore derived, localizer and 2D images')
    series_numbers = InputMultiPath(traits.Str(), argstr='-n %s...', desc='Selectively convert by series number - can be used up to 16 times')
    philips_float = traits.Bool(argstr='-p', desc='Philips precise float (not display) scaling')
    to_nrrd = traits.Bool(argstr='-e', desc='Export as NRRD instead of NIfTI')