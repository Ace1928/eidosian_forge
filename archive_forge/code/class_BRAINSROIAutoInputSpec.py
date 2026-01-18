from nipype.interfaces.base import (
import os
class BRAINSROIAutoInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='The input image for finding the largest region filled mask.', exists=True, argstr='--inputVolume %s')
    outputROIMaskVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='The ROI automatically found from the input image.', argstr='--outputROIMaskVolume %s')
    outputClippedVolumeROI = traits.Either(traits.Bool, File(), hash_files=False, desc='The inputVolume clipped to the region of the brain mask.', argstr='--outputClippedVolumeROI %s')
    otsuPercentileThreshold = traits.Float(desc='Parameter to the Otsu threshold algorithm.', argstr='--otsuPercentileThreshold %f')
    thresholdCorrectionFactor = traits.Float(desc="A factor to scale the Otsu algorithm's result threshold, in case clipping mangles the image.", argstr='--thresholdCorrectionFactor %f')
    closingSize = traits.Float(desc='The Closing Size (in millimeters) for largest connected filled mask.  This value is divided by image spacing and rounded to the next largest voxel number.', argstr='--closingSize %f')
    ROIAutoDilateSize = traits.Float(desc='This flag is only relevant when using ROIAUTO mode for initializing masks.  It defines the final dilation size to capture a bit of background outside the tissue region.  At setting of 10mm has been shown to help regularize a BSpline registration type so that there is some background constraints to match the edges of the head better.', argstr='--ROIAutoDilateSize %f')
    outputVolumePixelType = traits.Enum('float', 'short', 'ushort', 'int', 'uint', 'uchar', desc='The output image Pixel Type is the scalar datatype for representation of the Output Volume.', argstr='--outputVolumePixelType %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')