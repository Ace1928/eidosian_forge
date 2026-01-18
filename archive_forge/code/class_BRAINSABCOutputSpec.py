import os
from ...base import (
class BRAINSABCOutputSpec(TraitedSpec):
    saveState = File(desc='(optional) Filename to which save the final state of the registration', exists=True)
    outputDir = Directory(desc='Output directory', exists=True)
    atlasToSubjectTransform = File(desc='The transform from atlas to the subject', exists=True)
    atlasToSubjectInitialTransform = File(desc='The initial transform from atlas to the subject', exists=True)
    outputVolumes = OutputMultiPath(File(exists=True), desc='Corrected Output Images: should specify the same number of images as inputVolume, if only one element is given, then it is used as a file pattern where %s is replaced by the imageVolumeType, and %d by the index list location.')
    outputLabels = File(desc='Output Label Image', exists=True)
    outputDirtyLabels = File(desc='Output Dirty Label Image', exists=True)
    implicitOutputs = OutputMultiPath(File(exists=True), desc='Outputs to be made available to NiPype. Needed because not all BRAINSABC outputs have command line arguments.')