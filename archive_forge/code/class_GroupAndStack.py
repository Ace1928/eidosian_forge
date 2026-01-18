import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class GroupAndStack(DcmStack):
    """Create (potentially) multiple Nifti files for a set of DICOM files."""
    input_spec = DcmStackInputSpec
    output_spec = GroupAndStackOutputSpec

    def _run_interface(self, runtime):
        src_paths = self._get_filelist(self.inputs.dicom_files)
        stacks = dcmstack.parse_and_stack(src_paths)
        self.out_list = []
        for key, stack in list(stacks.items()):
            nw = NiftiWrapper(stack.to_nifti(embed_meta=True))
            const_meta = nw.meta_ext.get_class_dict(('global', 'const'))
            out_path = self._get_out_path(const_meta)
            if not self.inputs.embed_meta:
                nw.remove_extension()
            nb.save(nw.nii_img, out_path)
            self.out_list.append(out_path)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_list'] = self.out_list
        return outputs