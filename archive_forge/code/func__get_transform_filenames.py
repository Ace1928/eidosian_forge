import os
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, InputMultiObject
from ...utils.filemanip import split_filename
def _get_transform_filenames(self):
    retval = []
    for ii in range(len(self.inputs.transforms)):
        if isdefined(self.inputs.invert_transform_flags):
            if len(self.inputs.transforms) == len(self.inputs.invert_transform_flags):
                invert_code = 1 if self.inputs.invert_transform_flags[ii] else 0
                retval.append('--transform [ %s, %d ]' % (self.inputs.transforms[ii], invert_code))
            else:
                raise Exception('ERROR: The useInverse list must have the same number of entries as the transformsFileName list.')
        else:
            retval.append('--transform %s' % self.inputs.transforms[ii])
    return ' '.join(retval)