import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
class DTIReconOutputSpec(TraitedSpec):
    ADC = File(exists=True)
    B0 = File(exists=True)
    L1 = File(exists=True)
    L2 = File(exists=True)
    L3 = File(exists=True)
    exp = File(exists=True)
    FA = File(exists=True)
    FA_color = File(exists=True)
    tensor = File(exists=True)
    V1 = File(exists=True)
    V2 = File(exists=True)
    V3 = File(exists=True)