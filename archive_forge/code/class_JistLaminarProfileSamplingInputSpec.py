import os
from ..base import (
class JistLaminarProfileSamplingInputSpec(CommandLineInputSpec):
    inProfile = File(desc='Profile Surface Image', exists=True, argstr='--inProfile %s')
    inIntensity = File(desc='Intensity Image', exists=True, argstr='--inIntensity %s')
    inCortex = File(desc='Cortex Mask (opt)', exists=True, argstr='--inCortex %s')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outProfilemapped = traits.Either(traits.Bool, File(), hash_files=False, desc='Profile-mapped Intensity Image', argstr='--outProfilemapped %s')
    outProfile2 = traits.Either(traits.Bool, File(), hash_files=False, desc='Profile 4D Mask', argstr='--outProfile2 %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)