import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
class AFNICommandInputSpec(CommandLineInputSpec):
    num_threads = traits.Int(1, usedefault=True, nohash=True, desc='set number of threads')
    outputtype = traits.Enum('AFNI', list(Info.ftypes.keys()), desc='AFNI output filetype')
    out_file = File(name_template='%s_afni', desc='output image file name', argstr='-prefix %s', name_source=['in_file'])