from nipype.interfaces.base import (
import os
class ResampleDTIVolume(SEMLikeCommandLine):
    """title: Resample DTI Volume

    category: Diffusion.Diffusion Tensor Images

    description: Resampling an image is a very important task in image analysis. It is especially important in the frame of image registration. This module implements DT image resampling through the use of itk Transforms. The resampling is controlled by the Output Spacing. "Resampling" is performed in space coordinates, not pixel/grid coordinates. It is quite important to ensure that image spacing is properly set on the images involved. The interpolator is required since the mapping from one space to the other will often require evaluation of the intensity of the image at non-grid positions.

    version: 0.1

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/ResampleDTI

    contributor: Francois Budin (UNC)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149. Information on the National Centers for Biomedical Computing can be obtained from http://nihroadmap.nih.gov/bioinformatics
    """
    input_spec = ResampleDTIVolumeInputSpec
    output_spec = ResampleDTIVolumeOutputSpec
    _cmd = 'ResampleDTIVolume '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}