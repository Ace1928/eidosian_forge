import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _format_transform(self, index):
    retval = []
    retval.append('%s[ ' % self.inputs.transforms[index])
    parameters = ', '.join([str(element) for element in self.inputs.transform_parameters[index]])
    retval.append('%s' % parameters)
    retval.append(' ]')
    return ''.join(retval)