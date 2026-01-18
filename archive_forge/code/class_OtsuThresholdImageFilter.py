from nipype.interfaces.base import (
import os
class OtsuThresholdImageFilter(SEMLikeCommandLine):
    """title: Otsu Threshold Image Filter

    category: Legacy.Filtering

    description: This filter creates a binary thresholded image that separates an image into foreground and background components. The filter calculates the optimum threshold separating those two classes so that their combined spread (intra-class variance) is minimal (see http://en.wikipedia.org/wiki/Otsu%27s_method).  Then the filter applies that threshold to the input image using the itkBinaryThresholdImageFilter. The numberOfHistogram bins can be set for the Otsu Calculator. The insideValue and outsideValue can be set for the BinaryThresholdImageFilter.  The filter produces a labeled volume.

    The original reference is:

    N.Otsu, A threshold selection method from gray level histograms, IEEE Trans.Syst.ManCybern.SMC-9,62–66 1979.

    version: 0.1.0.$Revision: 19608 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/OtsuThresholdImageFilter

    contributor: Bill Lorensen (GE)

    acknowledgements: This command module was derived from Insight/Examples (copyright) Insight Software Consortium
    """
    input_spec = OtsuThresholdImageFilterInputSpec
    output_spec = OtsuThresholdImageFilterOutputSpec
    _cmd = 'OtsuThresholdImageFilter '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}