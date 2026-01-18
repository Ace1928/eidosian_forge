import os
from ..base import (
from ...external.due import BibTeX
from .base import (
class DeconvolveOutputSpec(TraitedSpec):
    out_file = File(desc='output statistics file', exists=True)
    reml_script = File(desc='automatically generated script to run 3dREMLfit', exists=True)
    x1D = File(desc='save out X matrix', exists=True)
    cbucket = File(desc='output regression coefficients file (if generated)')