import os
from ...utils.filemanip import split_filename
from ..base import CommandLineInputSpec, CommandLine, traits, TraitedSpec, File
class Trackvis2CaminoOutputSpec(TraitedSpec):
    camino = File(exists=True, desc='The filename to which to write the .Bfloat (camino).')