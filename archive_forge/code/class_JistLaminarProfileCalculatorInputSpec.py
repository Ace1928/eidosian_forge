import os
from ..base import (
class JistLaminarProfileCalculatorInputSpec(CommandLineInputSpec):
    inIntensity = File(desc='Intensity Profile Image', exists=True, argstr='--inIntensity %s')
    inMask = File(desc='Mask Image (opt, 3D or 4D)', exists=True, argstr='--inMask %s')
    incomputed = traits.Enum('mean', 'stdev', 'skewness', 'kurtosis', desc='computed statistic', argstr='--incomputed %s')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outResult = traits.Either(traits.Bool, File(), hash_files=False, desc='Result', argstr='--outResult %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)