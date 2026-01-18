import os
from ..base import (
class JistIntensityMp2rageMaskingInputSpec(CommandLineInputSpec):
    inSecond = File(desc='Second inversion (Inv2) Image', exists=True, argstr='--inSecond %s')
    inQuantitative = File(desc='Quantitative T1 Map (T1_Images) Image', exists=True, argstr='--inQuantitative %s')
    inT1weighted = File(desc='T1-weighted (UNI) Image', exists=True, argstr='--inT1weighted %s')
    inBackground = traits.Enum('exponential', 'half-normal', desc='Model distribution for background noise (default is half-normal, exponential is more stringent).', argstr='--inBackground %s')
    inSkip = traits.Enum('true', 'false', desc='Skip zero values', argstr='--inSkip %s')
    inMasking = traits.Enum('binary', 'proba', desc='Whether to use a binary threshold or a weighted average based on the probability.', argstr='--inMasking %s')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outSignal = traits.Either(traits.Bool, File(), hash_files=False, desc='Signal Proba Image', argstr='--outSignal_Proba %s')
    outSignal2 = traits.Either(traits.Bool, File(), hash_files=False, desc='Signal Mask Image', argstr='--outSignal_Mask %s')
    outMasked = traits.Either(traits.Bool, File(), hash_files=False, desc='Masked T1 Map Image', argstr='--outMasked_T1_Map %s')
    outMasked2 = traits.Either(traits.Bool, File(), hash_files=False, desc='Masked Iso Image', argstr='--outMasked_T1weighted %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)