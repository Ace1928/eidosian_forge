import os
from ...utils.filemanip import split_filename
from ..base import (
class SFPICOCalibDataOutputSpec(TraitedSpec):
    PICOCalib = File(exists=True, desc='Calibration dataset')
    calib_info = File(exists=True, desc='Calibration dataset')