import os
from ...utils.filemanip import split_filename
from ..base import (
class DTLUTGenOutputSpec(TraitedSpec):
    dtLUT = File(exists=True, desc='Lookup Table')