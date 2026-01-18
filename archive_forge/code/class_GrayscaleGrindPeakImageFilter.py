from nipype.interfaces.base import (
import os
class GrayscaleGrindPeakImageFilter(SEMLikeCommandLine):
    """title: Grayscale Grind Peak Image Filter

    category: Filtering.Morphology

    description: GrayscaleGrindPeakImageFilter removes peaks in a grayscale image. Peaks are local maxima in the grayscale topography that are not connected to boundaries of the image. Gray level values adjacent to a peak are extrapolated through the peak.

    This filter is used to smooth over local maxima without affecting the values of local minima.  If you take the difference between the output of this filter and the original image (and perhaps threshold the difference above a small value), you'll obtain a map of the local maxima.

    This filter uses the GrayscaleGeodesicDilateImageFilter.  It provides its own input as the "mask" input to the geodesic erosion.  The "marker" image for the geodesic erosion is constructed such that boundary pixels match the boundary pixels of the input image and the interior pixels are set to the minimum pixel value in the input image.

    This filter is the dual to the GrayscaleFillholeImageFilter which implements the Fillhole algorithm.  Since it is a dual, it is somewhat superfluous but is provided as a convenience.

    Geodesic morphology and the Fillhole algorithm is described in Chapter 6 of Pierre Soille's book "Morphological Image Analysis: Principles and Applications", Second Edition, Springer, 2003.

    A companion filter, Grayscale Fill Hole, fills holes in grayscale images.

    version: 0.1.0.$Revision: 19608 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/GrayscaleGrindPeakImageFilter

    contributor: Bill Lorensen (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = GrayscaleGrindPeakImageFilterInputSpec
    output_spec = GrayscaleGrindPeakImageFilterOutputSpec
    _cmd = 'GrayscaleGrindPeakImageFilter '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}