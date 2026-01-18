import os
from ...base import (
class insertMidACPCpoint(SEMLikeCommandLine):
    """title: MidACPC Landmark Insertion

    category: Utilities.BRAINS

    description: This program gets a landmark fcsv file and adds a new landmark as the midpoint between AC and PC points to the output landmark fcsv file

    contributor: Ali Ghayoor
    """
    input_spec = insertMidACPCpointInputSpec
    output_spec = insertMidACPCpointOutputSpec
    _cmd = ' insertMidACPCpoint '
    _outputs_filenames = {'outputLandmarkFile': 'outputLandmarkFile'}
    _redirect_x = False