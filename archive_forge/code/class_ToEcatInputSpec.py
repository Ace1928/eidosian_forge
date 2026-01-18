import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class ToEcatInputSpec(CommandLineInputSpec):
    input_file = File(desc='input file to convert', exists=True, mandatory=True, argstr='%s', position=-2)
    output_file = File(desc='output file', genfile=True, argstr='%s', position=-1, name_source=['input_file'], hash_files=False, name_template='%s_to_ecat.v', keep_extension=False)
    ignore_patient_variable = traits.Bool(desc='Ignore information from the minc patient variable.', argstr='-ignore_patient_variable')
    ignore_study_variable = traits.Bool(desc='Ignore information from the minc study variable.', argstr='-ignore_study_variable')
    ignore_acquisition_variable = traits.Bool(desc='Ignore information from the minc acquisition variable.', argstr='-ignore_acquisition_variable')
    ignore_ecat_acquisition_variable = traits.Bool(desc='Ignore information from the minc ecat_acquisition variable.', argstr='-ignore_ecat_acquisition_variable')
    ignore_ecat_main = traits.Bool(desc='Ignore information from the minc ecat-main variable.', argstr='-ignore_ecat_main')
    ignore_ecat_subheader_variable = traits.Bool(desc='Ignore information from the minc ecat-subhdr variable.', argstr='-ignore_ecat_subheader_variable')
    no_decay_corr_fctr = traits.Bool(desc='Do not compute the decay correction factors', argstr='-no_decay_corr_fctr')
    voxels_as_integers = traits.Bool(desc='Voxel values are treated as integers, scale andcalibration factors are set to unity', argstr='-label')