import os
from ...utils.filemanip import split_filename
from ..base import (
class SFLUTGenOutputSpec(TraitedSpec):
    lut_one_fibre = File(exists=True, desc='PICo lut for one-fibre model')
    lut_two_fibres = File(exists=True, desc='PICo lut for two-fibre model')