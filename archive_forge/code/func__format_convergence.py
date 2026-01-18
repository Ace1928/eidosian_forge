import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _format_convergence(self, ii):
    convergence_iter = self._format_xarray(self.inputs.number_of_iterations[ii])
    if len(self.inputs.convergence_threshold) > ii:
        convergence_value = self.inputs.convergence_threshold[ii]
    else:
        convergence_value = self.inputs.convergence_threshold[0]
    if len(self.inputs.convergence_window_size) > ii:
        convergence_ws = self.inputs.convergence_window_size[ii]
    else:
        convergence_ws = self.inputs.convergence_window_size[0]
    return '[ %s, %g, %d ]' % (convergence_iter, convergence_value, convergence_ws)