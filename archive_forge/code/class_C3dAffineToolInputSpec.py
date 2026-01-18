import os
from glob import glob
from .base import (
from ..utils.filemanip import split_filename
from .. import logging
class C3dAffineToolInputSpec(CommandLineInputSpec):
    reference_file = File(exists=True, argstr='-ref %s', position=1)
    source_file = File(exists=True, argstr='-src %s', position=2)
    transform_file = File(exists=True, argstr='%s', position=3)
    itk_transform = traits.Either(traits.Bool, File(), hash_files=False, desc='Export ITK transform.', argstr='-oitk %s', position=5)
    fsl2ras = traits.Bool(argstr='-fsl2ras', position=4)