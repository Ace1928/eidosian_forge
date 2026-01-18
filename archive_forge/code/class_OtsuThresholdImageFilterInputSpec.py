from nipype.interfaces.base import (
import os
class OtsuThresholdImageFilterInputSpec(CommandLineInputSpec):
    insideValue = traits.Int(desc='The value assigned to pixels that are inside the computed threshold', argstr='--insideValue %d')
    outsideValue = traits.Int(desc='The value assigned to pixels that are outside the computed threshold', argstr='--outsideValue %d')
    numberOfBins = traits.Int(desc='This is an advanced parameter. The number of bins in the histogram used to model the probability mass function of the two intensity distributions. Small numbers of bins may result in a more conservative threshold. The default should suffice for most applications. Experimentation is the only way to see the effect of varying this parameter.', argstr='--numberOfBins %d')
    inputVolume = File(position=-2, desc='Input volume to be filtered', exists=True, argstr='%s')
    outputVolume = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Output filtered', argstr='%s')