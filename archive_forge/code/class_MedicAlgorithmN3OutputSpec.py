import os
from ..base import (
class MedicAlgorithmN3OutputSpec(TraitedSpec):
    outInhomogeneity = File(desc='Inhomogeneity Corrected Volume', exists=True)
    outInhomogeneity2 = File(desc='Inhomogeneity Field', exists=True)