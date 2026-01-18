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
class PickAtlas(BaseInterface):
    """Returns ROI masks given an atlas and a list of labels. Supports dilation
    and left right masking (assuming the atlas is properly aligned).
    """
    input_spec = PickAtlasInputSpec
    output_spec = PickAtlasOutputSpec

    def _run_interface(self, runtime):
        nim = self._get_brodmann_area()
        nb.save(nim, self._gen_output_filename())
        return runtime

    def _gen_output_filename(self):
        if not isdefined(self.inputs.output_file):
            output = fname_presuffix(fname=self.inputs.atlas, suffix='_mask', newpath=os.getcwd(), use_ext=True)
        else:
            output = os.path.realpath(self.inputs.output_file)
        return output

    def _get_brodmann_area(self):
        nii = nb.load(self.inputs.atlas)
        origdata = np.asanyarray(nii.dataobj)
        newdata = np.zeros(origdata.shape)
        if not isinstance(self.inputs.labels, list):
            labels = [self.inputs.labels]
        else:
            labels = self.inputs.labels
        for lab in labels:
            newdata[origdata == lab] = 1
        if self.inputs.hemi == 'right':
            newdata[int(floor(float(origdata.shape[0]) / 2)):, :, :] = 0
        elif self.inputs.hemi == 'left':
            newdata[:int(ceil(float(origdata.shape[0]) / 2)), :, :] = 0
        if self.inputs.dilation_size != 0:
            from scipy.ndimage.morphology import grey_dilation
            newdata = grey_dilation(newdata, (2 * self.inputs.dilation_size + 1, 2 * self.inputs.dilation_size + 1, 2 * self.inputs.dilation_size + 1))
        return nb.Nifti1Image(newdata, nii.affine, nii.header)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['mask_file'] = self._gen_output_filename()
        return outputs