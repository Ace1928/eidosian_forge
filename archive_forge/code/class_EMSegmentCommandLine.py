from nipype.interfaces.base import (
import os
class EMSegmentCommandLine(SEMLikeCommandLine):
    """title:
      EMSegment Command-line


    category:
      Segmentation.Specialized


    description:
      This module is used to simplify the process of segmenting large collections of images by providing a command line interface to the EMSegment algorithm for script and batch processing.


    documentation-url: http://www.slicer.org/slicerWiki/index.php/Documentation/4.0/EMSegment_Command-line

    contributor: Sebastien Barre, Brad Davis, Kilian Pohl, Polina Golland, Yumin Yuan, Daniel Haehn

    acknowledgements: Many people and organizations have contributed to the funding, design, and development of the EMSegment algorithm and its various implementations.

    """
    input_spec = EMSegmentCommandLineInputSpec
    output_spec = EMSegmentCommandLineOutputSpec
    _cmd = 'EMSegmentCommandLine '
    _outputs_filenames = {'generateEmptyMRMLSceneAndQuit': 'generateEmptyMRMLSceneAndQuit', 'resultMRMLSceneFileName': 'resultMRMLSceneFileName', 'resultVolumeFileName': 'resultVolumeFileName.mhd'}