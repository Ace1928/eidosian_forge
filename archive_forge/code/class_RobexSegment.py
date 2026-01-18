import os
from pathlib import Path
from nipype.interfaces.base import (
from nipype.utils.filemanip import split_filename
class RobexSegment(CommandLine):
    """

    ROBEX is an automatic whole-brain extraction tool for T1-weighted MRI data (commonly known as skull stripping).
    ROBEX aims for robust skull-stripping across datasets with no parameter settings. It fits a triangular mesh,
    constrained by a shape model, to the probabilistic output of a supervised brain boundary classifier.
    Because the shape model cannot perfectly accommodate unseen cases, a small free deformation is subsequently allowed.
    The deformation is optimized using graph cuts.
    The method ROBEX is based on was published in IEEE Transactions on Medical Imaging;
    please visit the website http://www.jeiglesias.com to download the paper.

    Examples
    --------
    >>> from nipype.interfaces.robex.preprocess import RobexSegment
    >>> robex = RobexSegment()
    >>> robex.inputs.in_file = 'structural.nii'
    >>> robex.cmdline
    'runROBEX.sh structural.nii structural_brain.nii structural_brainmask.nii'
    >>> robex.run() # doctest: +SKIP

    """
    input_spec = RobexInputSpec
    output_spec = RobexOutputSpec
    _cmd = 'runROBEX.sh'