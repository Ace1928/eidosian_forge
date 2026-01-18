from nipype.interfaces.base import (
import os
class PETStandardUptakeValueComputationOutputSpec(TraitedSpec):
    csvFile = File(desc='A file holding the output SUV values in comma separated lines, one per label. Optional.', exists=True)