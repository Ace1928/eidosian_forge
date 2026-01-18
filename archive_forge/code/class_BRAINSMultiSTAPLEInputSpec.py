import os
from ...base import (
class BRAINSMultiSTAPLEInputSpec(CommandLineInputSpec):
    inputCompositeT1Volume = File(desc='Composite T1, all label maps transformed into the space for this image.', exists=True, argstr='--inputCompositeT1Volume %s')
    inputLabelVolume = InputMultiPath(File(exists=True), desc='The list of proobabilityimages.', argstr='--inputLabelVolume %s...')
    inputTransform = InputMultiPath(File(exists=True), desc='transforms to apply to label volumes', argstr='--inputTransform %s...')
    labelForUndecidedPixels = traits.Int(desc='Label for undecided pixels', argstr='--labelForUndecidedPixels %d')
    resampledVolumePrefix = traits.Str(desc='if given, write out resampled volumes with this prefix', argstr='--resampledVolumePrefix %s')
    skipResampling = traits.Bool(desc='Omit resampling images into reference space', argstr='--skipResampling ')
    outputMultiSTAPLE = traits.Either(traits.Bool, File(), hash_files=False, desc='the MultiSTAPLE average of input label volumes', argstr='--outputMultiSTAPLE %s')
    outputConfusionMatrix = traits.Either(traits.Bool, File(), hash_files=False, desc='Confusion Matrix', argstr='--outputConfusionMatrix %s')