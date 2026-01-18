import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class AddCSVColumn(BaseInterface):
    """
    Short interface to add an extra column and field to a text file.

    Example
    -------
    >>> from nipype.algorithms import misc
    >>> addcol = misc.AddCSVColumn()
    >>> addcol.inputs.in_file = 'degree.csv'
    >>> addcol.inputs.extra_column_heading = 'group'
    >>> addcol.inputs.extra_field = 'male'
    >>> addcol.run() # doctest: +SKIP

    """
    input_spec = AddCSVColumnInputSpec
    output_spec = AddCSVColumnOutputSpec

    def _run_interface(self, runtime):
        in_file = open(self.inputs.in_file, 'r')
        _, name, ext = split_filename(self.inputs.out_file)
        if not ext == '.csv':
            ext = '.csv'
        out_file = op.abspath(name + ext)
        out_file = open(out_file, 'w')
        firstline = in_file.readline()
        firstline = firstline.replace('\n', '')
        new_firstline = firstline + ',"' + self.inputs.extra_column_heading + '"\n'
        out_file.write(new_firstline)
        for line in in_file:
            new_line = line.replace('\n', '')
            new_line = new_line + ',' + self.inputs.extra_field + '\n'
            out_file.write(new_line)
        in_file.close()
        out_file.close()
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        _, name, ext = split_filename(self.inputs.out_file)
        if not ext == '.csv':
            ext = '.csv'
        out_file = op.abspath(name + ext)
        outputs['csv_file'] = out_file
        return outputs