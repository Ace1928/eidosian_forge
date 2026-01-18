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
class CreateNifti(BaseInterface):
    """Creates a nifti volume"""
    input_spec = CreateNiftiInputSpec
    output_spec = CreateNiftiOutputSpec

    def _gen_output_file_name(self):
        _, base, _ = split_filename(self.inputs.data_file)
        return os.path.abspath(base + '.nii')

    def _run_interface(self, runtime):
        with open(self.inputs.header_file, 'rb') as hdr_file:
            hdr = nb.AnalyzeHeader.from_fileobj(hdr_file)
        if isdefined(self.inputs.affine):
            affine = self.inputs.affine
        else:
            affine = None
        with open(self.inputs.data_file, 'rb') as data_file:
            data = hdr.data_from_fileobj(data_file)
        img = nb.Nifti1Image(data, affine, hdr)
        nb.save(img, self._gen_output_file_name())
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['nifti_file'] = self._gen_output_file_name()
        return outputs