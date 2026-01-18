import os
from ...base import (
class BRAINSCreateLabelMapFromProbabilityMapsInputSpec(CommandLineInputSpec):
    inputProbabilityVolume = InputMultiPath(File(exists=True), desc='The list of proobabilityimages.', argstr='--inputProbabilityVolume %s...')
    priorLabelCodes = InputMultiPath(traits.Int, desc='A list of PriorLabelCode values used for coding the output label images', sep=',', argstr='--priorLabelCodes %s')
    foregroundPriors = InputMultiPath(traits.Int, desc='A list: For each Prior Label, 1 if foreground, 0 if background', sep=',', argstr='--foregroundPriors %s')
    nonAirRegionMask = File(desc="a mask representing the 'NonAirRegion' -- Just force pixels in this region to zero", exists=True, argstr='--nonAirRegionMask %s')
    inclusionThreshold = traits.Float(desc='tolerance for inclusion', argstr='--inclusionThreshold %f')
    dirtyLabelVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='the labels prior to cleaning', argstr='--dirtyLabelVolume %s')
    cleanLabelVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='the foreground labels volume', argstr='--cleanLabelVolume %s')