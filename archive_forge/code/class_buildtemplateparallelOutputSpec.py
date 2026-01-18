from builtins import range
import os
from glob import glob
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, OutputMultiPath
from ...utils.filemanip import split_filename
class buildtemplateparallelOutputSpec(TraitedSpec):
    final_template_file = File(exists=True, desc='final ANTS template')
    template_files = OutputMultiPath(File(exists=True), desc='Templates from different stages of iteration')
    subject_outfiles = OutputMultiPath(File(exists=True), desc='Outputs for each input image. Includes warp field, inverse warp, Affine, original image (repaired) and warped image (deformed)')