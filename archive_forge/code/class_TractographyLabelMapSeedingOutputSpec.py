from nipype.interfaces.base import (
import os
class TractographyLabelMapSeedingOutputSpec(TraitedSpec):
    OutputFibers = File(position=-1, desc='Tractography result', exists=True)
    outputdirectory = Directory(desc='Directory in which to save fiber(s)', exists=True)