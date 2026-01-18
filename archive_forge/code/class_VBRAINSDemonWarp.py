from nipype.interfaces.base import (
import os
class VBRAINSDemonWarp(SEMLikeCommandLine):
    """title: Vector Demon Registration (BRAINS)

    category: Registration.Specialized

    description:
        This program finds a deformation field to warp a moving image onto a fixed image.  The images must be of the same signal kind, and contain an image of the same kind of object.  This program uses the Thirion Demons warp software in ITK, the Insight Toolkit.  Additional information is available at: http://www.nitrc.org/projects/brainsdemonwarp.



    version: 3.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:BRAINSDemonWarp

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool was developed by Hans J. Johnson and Greg Harris.

    acknowledgements: The development of this tool was supported by funding from grants NS050568 and NS40068 from the National Institute of Neurological Disorders and Stroke and grants MH31593, MH40856, from the National Institute of Mental Health.
    """
    input_spec = VBRAINSDemonWarpInputSpec
    output_spec = VBRAINSDemonWarpOutputSpec
    _cmd = 'VBRAINSDemonWarp '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii', 'outputCheckerboardVolume': 'outputCheckerboardVolume.nii', 'outputDisplacementFieldVolume': 'outputDisplacementFieldVolume.nrrd'}