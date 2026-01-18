import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class Dump(StdOutCommandLine):
    """Dump a MINC file. Typically used in conjunction with mincgen (see Gen).

    Examples
    --------

    >>> from nipype.interfaces.minc import Dump
    >>> from nipype.interfaces.minc.testdata import minc2Dfile

    >>> dump = Dump(input_file=minc2Dfile)
    >>> dump.run() # doctest: +SKIP

    >>> dump = Dump(input_file=minc2Dfile, output_file='/tmp/out.txt', precision=(3, 4))
    >>> dump.run() # doctest: +SKIP

    """
    input_spec = DumpInputSpec
    output_spec = DumpOutputSpec
    _cmd = 'mincdump'

    def _format_arg(self, name, spec, value):
        if name == 'precision':
            if isinstance(value, int):
                return '-p %d' % value
            elif isinstance(value, tuple) and isinstance(value[0], int) and isinstance(value[1], int):
                return '-p %d,%d' % (value[0], value[1])
            else:
                raise ValueError('Invalid precision argument: ' + str(value))
        return super(Dump, self)._format_arg(name, spec, value)