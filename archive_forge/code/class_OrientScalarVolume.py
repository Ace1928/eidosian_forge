from nipype.interfaces.base import (
import os
class OrientScalarVolume(SEMLikeCommandLine):
    """title: Orient Scalar Volume

    category: Converters

    description: Orients an output volume. Rearranges the slices in a volume according to the selected orientation. The slices are not interpolated. They are just reordered and/or permuted. The resulting volume will cover the original volume. NOTE: since Slicer takes into account the orientation of a volume, the re-oriented volume will not show any difference from the original volume, To see the difference, save the volume and display it with a system that either ignores the orientation of the image (e.g. Paraview) or displays individual images.

    version: 0.1.0.$Revision: 19608 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/OrientImage

    contributor: Bill Lorensen (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = OrientScalarVolumeInputSpec
    output_spec = OrientScalarVolumeOutputSpec
    _cmd = 'OrientScalarVolume '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}