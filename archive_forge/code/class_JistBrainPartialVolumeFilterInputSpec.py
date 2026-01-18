import os
from ..base import (
class JistBrainPartialVolumeFilterInputSpec(CommandLineInputSpec):
    inInput = File(desc='Input Image', exists=True, argstr='--inInput %s')
    inPV = traits.Enum('bright', 'dark', 'both', desc='Outputs the raw intensity values or a probability score for the partial volume regions.', argstr='--inPV %s')
    inoutput = traits.Enum('probability', 'intensity', desc='output', argstr='--inoutput %s')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outPartial = traits.Either(traits.Bool, File(), hash_files=False, desc='Partial Volume Image', argstr='--outPartial %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)