import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class DcmStack(NiftiGeneratorBase):
    """Create one Nifti file from a set of DICOM files. Can optionally embed
    meta data.

    Example
    -------

    >>> from nipype.interfaces.dcmstack import DcmStack
    >>> stacker = DcmStack()
    >>> stacker.inputs.dicom_files = 'path/to/series/'
    >>> stacker.run() # doctest: +SKIP
    >>> result.outputs.out_file # doctest: +SKIP
    '/path/to/cwd/sequence.nii.gz'
    """
    input_spec = DcmStackInputSpec
    output_spec = DcmStackOutputSpec

    def _get_filelist(self, trait_input):
        if isinstance(trait_input, (str, bytes)):
            if op.isdir(trait_input):
                return glob(op.join(trait_input, '*.dcm'))
            else:
                return glob(trait_input)
        return trait_input

    def _run_interface(self, runtime):
        src_paths = self._get_filelist(self.inputs.dicom_files)
        include_regexes = dcmstack.default_key_incl_res
        if isdefined(self.inputs.include_regexes):
            include_regexes += self.inputs.include_regexes
        exclude_regexes = dcmstack.default_key_excl_res
        if isdefined(self.inputs.exclude_regexes):
            exclude_regexes += self.inputs.exclude_regexes
        meta_filter = dcmstack.make_key_regex_filter(exclude_regexes, include_regexes)
        stack = dcmstack.DicomStack(meta_filter=meta_filter)
        for src_path in src_paths:
            if not imghdr.what(src_path) == 'gif':
                src_dcm = pydicom.dcmread(src_path, force=self.inputs.force_read)
                stack.add_dcm(src_dcm)
        nii = stack.to_nifti(embed_meta=True)
        nw = NiftiWrapper(nii)
        self.out_path = self._get_out_path(nw.meta_ext.get_class_dict(('global', 'const')))
        if not self.inputs.embed_meta:
            nw.remove_extension()
        nb.save(nii, self.out_path)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self.out_path
        return outputs