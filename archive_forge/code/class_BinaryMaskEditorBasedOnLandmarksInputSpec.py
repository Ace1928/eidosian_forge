import os
from ...base import (
class BinaryMaskEditorBasedOnLandmarksInputSpec(CommandLineInputSpec):
    inputBinaryVolume = File(desc='Input binary image in which to be edited', exists=True, argstr='--inputBinaryVolume %s')
    outputBinaryVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Output binary image in which to be edited', argstr='--outputBinaryVolume %s')
    inputLandmarksFilename = File(desc=' The filename for the  landmark definition file in the same format produced by Slicer3 (.fcsv). ', exists=True, argstr='--inputLandmarksFilename %s')
    inputLandmarkNames = InputMultiPath(traits.Str, desc=' A target input landmark name to be edited. This should be listed in the inputLandmakrFilename Given. ', sep=',', argstr='--inputLandmarkNames %s')
    setCutDirectionForLandmark = InputMultiPath(traits.Str, desc='Setting the cutting out direction of the input binary image to the one of anterior, posterior, left, right, superior or posterior. (ENUMERATION: ANTERIOR, POSTERIOR, LEFT, RIGHT, SUPERIOR, POSTERIOR) ', sep=',', argstr='--setCutDirectionForLandmark %s')
    setCutDirectionForObliquePlane = InputMultiPath(traits.Str, desc='If this is true, the mask will be thresholded out to the direction of inferior, posterior,  and/or left. Default behavrior is that cutting out to the direction of superior, anterior and/or right. ', sep=',', argstr='--setCutDirectionForObliquePlane %s')
    inputLandmarkNamesForObliquePlane = InputMultiPath(traits.Str, desc=' Three subset landmark names of inputLandmarksFilename for a oblique plane computation. The plane computed for binary volume editing. ', sep=',', argstr='--inputLandmarkNamesForObliquePlane %s')