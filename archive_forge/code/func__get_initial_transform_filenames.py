import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _get_initial_transform_filenames(self):
    n_transforms = len(self.inputs.initial_moving_transform)
    invert_flags = [0] * n_transforms
    if isdefined(self.inputs.invert_initial_moving_transform):
        if len(self.inputs.invert_initial_moving_transform) != n_transforms:
            raise Exception('Inputs "initial_moving_transform" and "invert_initial_moving_transform"should have the same length.')
        invert_flags = self.inputs.invert_initial_moving_transform
    retval = ['[ %s, %d ]' % (xfm, int(flag)) for xfm, flag in zip(self.inputs.initial_moving_transform, invert_flags)]
    return ' '.join(['--initial-moving-transform'] + retval)