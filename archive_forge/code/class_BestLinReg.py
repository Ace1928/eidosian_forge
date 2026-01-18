import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class BestLinReg(CommandLine):
    """Hierachial linear fitting between two files.

    The bestlinreg script is part of the EZminc package:

    https://github.com/BIC-MNI/EZminc/blob/master/scripts/bestlinreg.pl

    Examples
    --------

    >>> from nipype.interfaces.minc import BestLinReg
    >>> from nipype.interfaces.minc.testdata import nonempty_minc_data

    >>> input_file = nonempty_minc_data(0)
    >>> target_file = nonempty_minc_data(1)
    >>> linreg = BestLinReg(source=input_file, target=target_file)
    >>> linreg.run() # doctest: +SKIP
    """
    input_spec = BestLinRegInputSpec
    output_spec = BestLinRegOutputSpec
    _cmd = 'bestlinreg'