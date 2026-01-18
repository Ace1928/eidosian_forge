import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class Beast(CommandLine):
    """Extract brain image using BEaST (Brain Extraction using
    non-local Segmentation Technique).

    Examples
    --------

    >>> from nipype.interfaces.minc import Beast
    >>> from nipype.interfaces.minc.testdata import nonempty_minc_data

    >>> file0 = nonempty_minc_data(0)
    >>> beast = Beast(input_file=file0)
    >>> beast .run() # doctest: +SKIP
    """
    input_spec = BeastInputSpec
    output_spec = BeastOutputSpec
    _cmd = 'mincbeast'