import os
from ...base import (
class gtractResampleFibers(SEMLikeCommandLine):
    """title: Resample Fibers

    category: Diffusion.GTRACT

    description: This program will resample a fiber tract with respect to a pair of deformation fields that represent the forward and reverse deformation fields.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractResampleFibersInputSpec
    output_spec = gtractResampleFibersOutputSpec
    _cmd = ' gtractResampleFibers '
    _outputs_filenames = {'outputTract': 'outputTract.vtk'}
    _redirect_x = False