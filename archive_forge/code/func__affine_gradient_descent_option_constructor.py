import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _affine_gradient_descent_option_constructor(self):
    values = self.inputs.affine_gradient_descent_option
    defaults = [0.1, 0.5, 0.0001, 0.0001]
    for ii in range(len(defaults)):
        try:
            defaults[ii] = values[ii]
        except IndexError:
            break
    parameters = self._format_xarray(['%g' % defaults[index] for index in range(4)])
    retval = ['--affine-gradient-descent-option', parameters]
    return ' '.join(retval)