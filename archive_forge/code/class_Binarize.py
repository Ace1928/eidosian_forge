import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class Binarize(FSCommand):
    """Use FreeSurfer mri_binarize to threshold an input volume

    Examples
    --------
    >>> binvol = Binarize(in_file='structural.nii', min=10, binary_file='foo_out.nii')
    >>> binvol.cmdline
    'mri_binarize --o foo_out.nii --i structural.nii --min 10.000000'

    """
    _cmd = 'mri_binarize'
    input_spec = BinarizeInputSpec
    output_spec = BinarizeOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outfile = self.inputs.binary_file
        if not isdefined(outfile):
            if isdefined(self.inputs.out_type):
                outfile = fname_presuffix(self.inputs.in_file, newpath=os.getcwd(), suffix='.'.join(('_thresh', self.inputs.out_type)), use_ext=False)
            else:
                outfile = fname_presuffix(self.inputs.in_file, newpath=os.getcwd(), suffix='_thresh')
        outputs['binary_file'] = os.path.abspath(outfile)
        value = self.inputs.count_file
        if isdefined(value):
            if isinstance(value, bool):
                if value:
                    outputs['count_file'] = fname_presuffix(self.inputs.in_file, suffix='_count.txt', newpath=os.getcwd(), use_ext=False)
            else:
                outputs['count_file'] = value
        return outputs

    def _format_arg(self, name, spec, value):
        if name == 'count_file':
            if isinstance(value, bool):
                fname = self._list_outputs()[name]
            else:
                fname = value
            return spec.argstr % fname
        if name == 'out_type':
            return ''
        return super(Binarize, self)._format_arg(name, spec, value)

    def _gen_filename(self, name):
        if name == 'binary_file':
            return self._list_outputs()[name]
        return None