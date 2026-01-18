import os
from ..base import (
class MedicAlgorithmImageCalculatorOutputSpec(TraitedSpec):
    outResult = File(desc='Result Volume', exists=True)