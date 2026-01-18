import os
from ..base import (
class MedicAlgorithmImageCalculatorInputSpec(CommandLineInputSpec):
    inVolume = File(desc='Volume 1', exists=True, argstr='--inVolume %s')
    inVolume2 = File(desc='Volume 2', exists=True, argstr='--inVolume2 %s')
    inOperation = traits.Enum('Add', 'Subtract', 'Multiply', 'Divide', 'Min', 'Max', desc='Operation', argstr='--inOperation %s')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outResult = traits.Either(traits.Bool, File(), hash_files=False, desc='Result Volume', argstr='--outResult %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)