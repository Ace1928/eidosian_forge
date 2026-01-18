import os
from ....base import (
class UKFTractographyOutputSpec(TraitedSpec):
    tracts = File(desc='Tracts generated, with first tensor output', exists=True)
    tractsWithSecondTensor = File(desc='Tracts generated, with second tensor output (if there is one)', exists=True)