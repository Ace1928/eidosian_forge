import numpy as np
from .analyze import AnalyzeHeader
from .batteryrunners import Report
from .filebasedimages import ImageFileError
from .nifti1 import Nifti1Header, Nifti1Image, Nifti1Pair
from .spatialimages import HeaderDataError
class Nifti2PairHeader(Nifti2Header):
    """Class for NIfTI2 pair header"""
    is_single = False