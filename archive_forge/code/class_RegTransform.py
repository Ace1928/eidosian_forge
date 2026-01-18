import os
from ..base import TraitedSpec, File, traits, isdefined
from .base import get_custom_path, NiftyRegCommand, NiftyRegCommandInputSpec
from ...utils.filemanip import split_filename
class RegTransform(NiftyRegCommand):
    """Interface for executable reg_transform from NiftyReg platform.

    Tools to convert transformation parametrisation from one type to another
    as well as to compose, inverse or half transformations.

    `Source code <https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg>`_

    Examples
    --------
    >>> from nipype.interfaces import niftyreg
    >>> node = niftyreg.RegTransform()
    >>> node.inputs.def_input = 'warpfield.nii'
    >>> node.inputs.omp_core_val = 4
    >>> node.cmdline  # doctest: +ELLIPSIS
    'reg_transform -omp 4 -def warpfield.nii .../warpfield_trans.nii.gz'

    """
    _cmd = get_custom_path('reg_transform')
    input_spec = RegTransformInputSpec
    output_spec = RegTransformOutputSpec
    _suffix = '_trans'

    def _find_input(self):
        inputs = [self.inputs.def_input, self.inputs.disp_input, self.inputs.flow_input, self.inputs.comp_input, self.inputs.comp_input2, self.inputs.upd_s_form_input, self.inputs.inv_aff_input, self.inputs.inv_nrr_input, self.inputs.half_input, self.inputs.make_aff_input, self.inputs.aff_2_rig_input, self.inputs.flirt_2_nr_input]
        entries = []
        for entry in inputs:
            if isdefined(entry):
                entries.append(entry)
                _, _, ext = split_filename(entry)
                if ext == '.nii' or ext == '.nii.gz' or ext == '.hdr':
                    return entry
        if len(entries):
            return entries[0]
        return None

    def _gen_filename(self, name):
        if name == 'out_file':
            if isdefined(self.inputs.make_aff_input):
                return self._gen_fname('matrix', suffix=self._suffix, ext='.txt')
            if isdefined(self.inputs.comp_input) and isdefined(self.inputs.comp_input2):
                _, bn1, ext1 = split_filename(self.inputs.comp_input)
                _, _, ext2 = split_filename(self.inputs.comp_input2)
                if ext1 in ['.nii', '.nii.gz', '.hdr', '.img', '.img.gz'] or ext2 in ['.nii', '.nii.gz', '.hdr', '.img', '.img.gz']:
                    return self._gen_fname(bn1, suffix=self._suffix, ext='.nii.gz')
                else:
                    return self._gen_fname(bn1, suffix=self._suffix, ext=ext1)
            if isdefined(self.inputs.flirt_2_nr_input):
                return self._gen_fname(self.inputs.flirt_2_nr_input[0], suffix=self._suffix, ext='.txt')
            input_to_use = self._find_input()
            _, _, ext = split_filename(input_to_use)
            if ext not in ['.nii', '.nii.gz', '.hdr', '.img', '.img.gz']:
                return self._gen_fname(input_to_use, suffix=self._suffix, ext=ext)
            else:
                return self._gen_fname(input_to_use, suffix=self._suffix, ext='.nii.gz')
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs['out_file'] = self.inputs.out_file
        else:
            outputs['out_file'] = self._gen_filename('out_file')
        return outputs