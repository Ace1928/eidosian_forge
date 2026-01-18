import os
from ...base import (
class gtractCoRegAnatomy(SEMLikeCommandLine):
    """title: Coregister B0 to Anatomy B-Spline

    category: Diffusion.GTRACT

    description: This program will register a Nrrd diffusion weighted 4D vector image to a fixed anatomical image. Two registration methods are supported for alignment with anatomical images: Rigid and B-Spline. The rigid registration performs a rigid body registration with the anatomical images and should be done as well to initialize the B-Spline transform. The B-SPline transform is the deformable transform, where the user can control the amount of deformation based on the number of control points as well as the maximum distance that these points can move. The B-Spline registration places a low dimensional grid in the image, which is deformed. This allows for some susceptibility related distortions to be removed from the diffusion weighted images. In general the amount of motion in the slice selection and read-out directions direction should be kept low. The distortion is in the phase encoding direction in the images. It is recommended that skull stripped (i.e. image containing only brain with skull removed) images should be used for image co-registration with the B-Spline transform.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractCoRegAnatomyInputSpec
    output_spec = gtractCoRegAnatomyOutputSpec
    _cmd = ' gtractCoRegAnatomy '
    _outputs_filenames = {'outputTransformName': 'outputTransformName.h5'}
    _redirect_x = False