import os
from ...utils.filemanip import split_filename
from ..base import (
class SFPeaksOutputSpec(TraitedSpec):
    peaks = File(exists=True, desc='Peaks of the spherical functions.')