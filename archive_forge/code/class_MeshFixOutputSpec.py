import os.path as op
from ..utils.filemanip import split_filename
from .base import (
class MeshFixOutputSpec(TraitedSpec):
    mesh_file = File(exists=True, desc='The output mesh file')