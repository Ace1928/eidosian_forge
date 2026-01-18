import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _mask_constructor(self):
    if self.inputs.moving_image_mask:
        retval = '--masks ["{fixed_image_mask}","{moving_image_mask}"]'.format(fixed_image_mask=self.inputs.fixed_image_mask, moving_image_mask=self.inputs.moving_image_mask)
    else:
        retval = '--masks "{fixed_image_mask}"'.format(fixed_image_mask=self.inputs.fixed_image_mask)
    return retval