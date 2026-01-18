import os
import re as regex
from ..base import (
class DewispOutputSpec(TraitedSpec):
    outputMaskFile = File(desc='path/name of mask file')