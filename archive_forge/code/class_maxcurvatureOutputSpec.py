import os
from ...base import (
class maxcurvatureOutputSpec(TraitedSpec):
    output = File(desc='Output File', exists=True)