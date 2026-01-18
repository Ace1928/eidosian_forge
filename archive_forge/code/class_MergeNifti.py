import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class MergeNifti(NiftiGeneratorBase):
    """Merge multiple Nifti files into one. Merges together meta data
    extensions as well."""
    input_spec = MergeNiftiInputSpec
    output_spec = MergeNiftiOutputSpec

    def _run_interface(self, runtime):
        niis = [nb.load(fn) for fn in self.inputs.in_files]
        nws = [NiftiWrapper(nii, make_empty=True) for nii in niis]
        if self.inputs.sort_order:
            sort_order = self.inputs.sort_order
            if isinstance(sort_order, (str, bytes)):
                sort_order = [sort_order]
            nws.sort(key=make_key_func(sort_order))
        if self.inputs.merge_dim == traits.Undefined:
            merge_dim = None
        else:
            merge_dim = self.inputs.merge_dim
        merged = NiftiWrapper.from_sequence(nws, merge_dim)
        const_meta = merged.meta_ext.get_class_dict(('global', 'const'))
        self.out_path = self._get_out_path(const_meta)
        nb.save(merged.nii_img, self.out_path)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_path
        return outputs