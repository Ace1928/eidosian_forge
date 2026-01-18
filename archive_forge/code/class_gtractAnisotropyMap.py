import os
from ...base import (
class gtractAnisotropyMap(SEMLikeCommandLine):
    """title: Anisotropy Map

    category: Diffusion.GTRACT

    description: This program will generate a scalar map of anisotropy, given a tensor representation. Anisotropy images are used for fiber tracking, but the anisotropy scalars are not defined along the path. Instead, the tensor representation is included as point data allowing all of these metrics to be computed using only the fiber tract point data. The images can be saved in any ITK supported format, but it is suggested that you use an image format that supports the definition of the image origin. This includes NRRD, NifTI, and Meta formats. These images can also be used for scalar analysis including regional anisotropy measures or VBM style analysis.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractAnisotropyMapInputSpec
    output_spec = gtractAnisotropyMapOutputSpec
    _cmd = ' gtractAnisotropyMap '
    _outputs_filenames = {'outputVolume': 'outputVolume.nrrd'}
    _redirect_x = False