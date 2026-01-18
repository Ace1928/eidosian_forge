import os.path as op
from ..base import traits, TraitedSpec, File
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class TractographyOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output filtered tracks')
    out_seeds = File(desc='output the seed location of all successful streamlines to a file')