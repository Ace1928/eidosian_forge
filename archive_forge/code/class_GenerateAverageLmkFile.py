from nipype.interfaces.base import (
import os
class GenerateAverageLmkFile(SEMLikeCommandLine):
    """title: Average Fiducials

    category: Testing

    description: This program gets several fcsv file each one contains several landmarks with the same name but slightly different coordinates. For EACH landmark we compute the average coordination.

    contributor: Ali Ghayoor
    """
    input_spec = GenerateAverageLmkFileInputSpec
    output_spec = GenerateAverageLmkFileOutputSpec
    _cmd = ' GenerateAverageLmkFile '
    _outputs_filenames = {'outputLandmarkFile': 'outputLandmarkFile'}
    _redirect_x = False