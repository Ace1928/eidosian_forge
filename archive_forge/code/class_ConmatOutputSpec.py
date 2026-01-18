import os
from ...utils.filemanip import split_filename
from ..base import (
class ConmatOutputSpec(TraitedSpec):
    conmat_sc = File(exists=True, desc='Connectivity matrix in CSV file.')
    conmat_ts = File(desc='Tract statistics in CSV file.')