import os
from ...utils.filemanip import split_filename
from ..base import (
class LinReconOutputSpec(TraitedSpec):
    recon_data = File(exists=True, desc='Transformed data')