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
class ModifyAffine(BaseInterface):
    """Left multiplies the affine matrix with a specified values. Saves the volume
    as a nifti file.
    """
    input_spec = ModifyAffineInputSpec
    output_spec = ModifyAffineOutputSpec

    def _gen_output_filename(self, name):
        _, base, _ = split_filename(name)
        return os.path.abspath(base + '_transformed.nii')

    def _run_interface(self, runtime):
        for fname in self.inputs.volumes:
            img = nb.load(fname)
            affine = img.affine
            affine = np.dot(self.inputs.transformation_matrix, affine)
            nb.save(nb.Nifti1Image(img.dataobj, affine, img.header), self._gen_output_filename(fname))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['transformed_volumes'] = []
        for fname in self.inputs.volumes:
            outputs['transformed_volumes'].append(self._gen_output_filename(fname))
        return outputs