import os
import re as regex
from ..base import (
class PvcOutputSpec(TraitedSpec):
    outputLabelFile = File(desc='path/name of label file')
    outputTissueFractionFile = File(desc='path/name of tissue fraction file')