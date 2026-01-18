import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class Volcentre(CommandLine):
    """Centre a MINC image's sampling about a point, typically (0,0,0).

    Example
    --------

    >>> from nipype.interfaces.minc import Volcentre
    >>> from nipype.interfaces.minc.testdata import minc2Dfile
    >>> vc = Volcentre(input_file=minc2Dfile)
    >>> vc.run() # doctest: +SKIP
    """
    input_spec = VolcentreInputSpec
    output_spec = VolcentreOutputSpec
    _cmd = 'volcentre'