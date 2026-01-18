import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class TCK2VTKOutputSpec(TraitedSpec):
    out_file = File(desc='output VTK file')