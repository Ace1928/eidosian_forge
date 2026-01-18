import os
from copy import deepcopy
from nibabel import load, funcs, Nifti1Image
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list, save_json, split_filename
from ..utils.misc import find_indices, normalize_mc_params
from .. import logging, config
def _get_output_filenames(self, motionfile, output_dir):
    """Generate output files based on motion filenames

        Parameters
        ----------
        motionfile: file/string
            Filename for motion parameter file
        output_dir: string
            output directory in which the files will be generated
        """
    _, filename = os.path.split(motionfile)
    filename, _ = os.path.splitext(filename)
    corrfile = os.path.join(output_dir, ''.join(('qa.', filename, '_stimcorr.txt')))
    return corrfile