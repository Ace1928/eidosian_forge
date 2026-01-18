from nipype.interfaces.base import (
import os
class DWIJointRicianLMMSEFilterOutputSpec(TraitedSpec):
    outputVolume = File(position=-1, desc='Output DWI volume.', exists=True)