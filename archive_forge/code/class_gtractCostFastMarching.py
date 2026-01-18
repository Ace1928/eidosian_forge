import os
from ...base import (
class gtractCostFastMarching(SEMLikeCommandLine):
    """title: Cost Fast Marching

    category: Diffusion.GTRACT

    description: This program will use a fast marching fiber tracking algorithm to identify fiber tracts from a tensor image. This program is the first portion of the algorithm. The user must first run gtractFastMarchingTracking to generate the actual fiber tracts.  This algorithm is roughly based on the work by G. Parker et al. from IEEE Transactions On Medical Imaging, 21(5): 505-512, 2002. An additional feature of including anisotropy into the vcl_cost function calculation is included.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris. The original code here was developed by Daisy Espino.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractCostFastMarchingInputSpec
    output_spec = gtractCostFastMarchingOutputSpec
    _cmd = ' gtractCostFastMarching '
    _outputs_filenames = {'outputCostVolume': 'outputCostVolume.nrrd', 'outputSpeedVolume': 'outputSpeedVolume.nrrd'}
    _redirect_x = False