import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class ToRaw(StdOutCommandLine):
    """Dump a chunk of MINC file data. This program is largely
    superseded by mincextract (see Extract).

    Examples
    --------

    >>> from nipype.interfaces.minc import ToRaw
    >>> from nipype.interfaces.minc.testdata import minc2Dfile

    >>> toraw = ToRaw(input_file=minc2Dfile)
    >>> toraw.run() # doctest: +SKIP

    >>> toraw = ToRaw(input_file=minc2Dfile, write_range=(0, 100))
    >>> toraw.run() # doctest: +SKIP
    """
    input_spec = ToRawInputSpec
    output_spec = ToRawOutputSpec
    _cmd = 'minctoraw'