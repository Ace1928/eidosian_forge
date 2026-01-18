import os
from ...utils.filemanip import split_filename
from ..base import (
class TrackOutputSpec(TraitedSpec):
    tracked = File(exists=True, desc='output file containing reconstructed tracts')