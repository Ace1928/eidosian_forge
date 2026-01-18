import os
from ...base import (
class gtractFastMarchingTracking(SEMLikeCommandLine):
    """title: Fast Marching Tracking

    category: Diffusion.GTRACT

    description: This program will use a fast marching fiber tracking algorithm to identify fiber tracts from a tensor image. This program is the second portion of the algorithm. The user must first run gtractCostFastMarching to generate the vcl_cost image. The second step of the algorithm implemented here is a gradient descent soplution from the defined ending region back to the seed points specified in gtractCostFastMarching. This algorithm is roughly based on the work by G. Parker et al. from IEEE Transactions On Medical Imaging, 21(5): 505-512, 2002. An additional feature of including anisotropy into the vcl_cost function calculation is included.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris. The original code here was developed by Daisy Espino.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractFastMarchingTrackingInputSpec
    output_spec = gtractFastMarchingTrackingOutputSpec
    _cmd = ' gtractFastMarchingTracking '
    _outputs_filenames = {'outputTract': 'outputTract.vtk'}
    _redirect_x = False