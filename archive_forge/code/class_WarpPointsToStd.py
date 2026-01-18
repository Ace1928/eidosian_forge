import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class WarpPointsToStd(WarpPoints):
    """
    Use FSL `img2stdcoord <http://fsl.fmrib.ox.ac.uk/fsl/fsl-4.1.9/flirt/overview.html>`_
    to transform point sets to standard space coordinates. Accepts plain text
    files and vtk files.

    .. Note:: transformation of TrackVis trk files is not yet implemented


    Examples
    --------

    >>> from nipype.interfaces.fsl import WarpPointsToStd
    >>> warppoints = WarpPointsToStd()
    >>> warppoints.inputs.in_coords = 'surf.txt'
    >>> warppoints.inputs.img_file = 'T1.nii'
    >>> warppoints.inputs.std_file = 'mni.nii'
    >>> warppoints.inputs.warp_file = 'warpfield.nii'
    >>> warppoints.inputs.coord_mm = True
    >>> warppoints.cmdline # doctest: +ELLIPSIS
    'img2stdcoord -mm -img T1.nii -std mni.nii -warp warpfield.nii surf.txt'
    >>> res = warppoints.run() # doctest: +SKIP


    """
    input_spec = WarpPointsToStdInputSpec
    output_spec = WarpPointsOutputSpec
    _cmd = 'img2stdcoord'
    _terminal_output = 'file_split'