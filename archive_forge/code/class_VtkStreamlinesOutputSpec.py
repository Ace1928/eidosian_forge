import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class VtkStreamlinesOutputSpec(TraitedSpec):
    vtk = File(exists=True, desc='Streamlines in VTK format')