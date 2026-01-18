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
class Matlab2CSV(BaseInterface):
    """
    Save the components of a MATLAB .mat file as a text file with comma-separated values (CSVs).

    CSV files are easily loaded in R, for use in statistical processing.
    For further information, see cran.r-project.org/doc/manuals/R-data.pdf

    Example
    -------
    >>> from nipype.algorithms import misc
    >>> mat2csv = misc.Matlab2CSV()
    >>> mat2csv.inputs.in_file = 'cmatrix.mat'
    >>> mat2csv.run() # doctest: +SKIP

    """
    input_spec = Matlab2CSVInputSpec
    output_spec = Matlab2CSVOutputSpec

    def _run_interface(self, runtime):
        import scipy.io as sio
        in_dict = sio.loadmat(op.abspath(self.inputs.in_file))
        saved_variables = list()
        for key in list(in_dict.keys()):
            if not key.startswith('__'):
                if isinstance(in_dict[key][0], np.ndarray):
                    saved_variables.append(key)
                else:
                    iflogger.info('One of the keys in the input file, %s, is not a Numpy array', key)
        if len(saved_variables) > 1:
            iflogger.info('%i variables found:', len(saved_variables))
            iflogger.info(saved_variables)
            for variable in saved_variables:
                iflogger.info('...Converting %s - type %s - to CSV', variable, type(in_dict[variable]))
                _matlab2csv(in_dict[variable], variable, self.inputs.reshape_matrix)
        elif len(saved_variables) == 1:
            _, name, _ = split_filename(self.inputs.in_file)
            variable = saved_variables[0]
            iflogger.info('Single variable found %s, type %s:', variable, type(in_dict[variable]))
            iflogger.info('...Converting %s to CSV from %s', variable, self.inputs.in_file)
            _matlab2csv(in_dict[variable], name, self.inputs.reshape_matrix)
        else:
            iflogger.error('No values in the MATLAB file?!')
        return runtime

    def _list_outputs(self):
        import scipy.io as sio
        outputs = self.output_spec().get()
        in_dict = sio.loadmat(op.abspath(self.inputs.in_file))
        saved_variables = list()
        for key in list(in_dict.keys()):
            if not key.startswith('__'):
                if isinstance(in_dict[key][0], np.ndarray):
                    saved_variables.append(key)
                else:
                    iflogger.error('One of the keys in the input file, %s, is not a Numpy array', key)
        if len(saved_variables) > 1:
            outputs['csv_files'] = replaceext(saved_variables, '.csv')
        elif len(saved_variables) == 1:
            _, name, ext = split_filename(self.inputs.in_file)
            outputs['csv_files'] = op.abspath(name + '.csv')
        else:
            iflogger.error('No values in the MATLAB file?!')
        return outputs