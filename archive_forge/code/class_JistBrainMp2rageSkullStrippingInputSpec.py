import os
from ..base import (
class JistBrainMp2rageSkullStrippingInputSpec(CommandLineInputSpec):
    inSecond = File(desc='Second inversion (Inv2) Image', exists=True, argstr='--inSecond %s')
    inT1 = File(desc='T1 Map (T1_Images) Image (opt)', exists=True, argstr='--inT1 %s')
    inT1weighted = File(desc='T1-weighted (UNI) Image (opt)', exists=True, argstr='--inT1weighted %s')
    inFilter = File(desc='Filter Image (opt)', exists=True, argstr='--inFilter %s')
    inSkip = traits.Enum('true', 'false', desc='Skip zero values', argstr='--inSkip %s')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outBrain = traits.Either(traits.Bool, File(), hash_files=False, desc='Brain Mask Image', argstr='--outBrain %s')
    outMasked = traits.Either(traits.Bool, File(), hash_files=False, desc='Masked T1 Map Image', argstr='--outMasked %s')
    outMasked2 = traits.Either(traits.Bool, File(), hash_files=False, desc='Masked T1-weighted Image', argstr='--outMasked2 %s')
    outMasked3 = traits.Either(traits.Bool, File(), hash_files=False, desc='Masked Filter Image', argstr='--outMasked3 %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)