import os
from ...base import (
class fcsv_to_hdf5(SEMLikeCommandLine):
    """title: fcsv_to_hdf5 (BRAINS)

    category: Utilities.BRAINS

    description: Convert a collection of fcsv files to a HDF5 format file
    """
    input_spec = fcsv_to_hdf5InputSpec
    output_spec = fcsv_to_hdf5OutputSpec
    _cmd = ' fcsv_to_hdf5 '
    _outputs_filenames = {'modelFile': 'modelFile', 'landmarksInformationFile': 'landmarksInformationFile.h5'}
    _redirect_x = False