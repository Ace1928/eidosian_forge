import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _image_metric_constructor(self):
    retval = []
    intensity_based = ['CC', 'MI', 'SMI', 'PR', 'SSD', 'MSQ']
    point_set_based = ['PSE', 'JTB']
    for ii in range(len(self.inputs.moving_image)):
        if self.inputs.metric[ii] in intensity_based:
            retval.append('--image-metric %s[ %s, %s, %g, %d ]' % (self.inputs.metric[ii], self.inputs.fixed_image[ii], self.inputs.moving_image[ii], self.inputs.metric_weight[ii], self.inputs.radius[ii]))
        elif self.inputs.metric[ii] == point_set_based:
            pass
    return ' '.join(retval)