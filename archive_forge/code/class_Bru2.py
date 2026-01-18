import os
from .base import (
class Bru2(CommandLine):
    """Uses bru2nii's Bru2 to convert Bruker files

    Examples
    ========

    >>> from nipype.interfaces.bru2nii import Bru2
    >>> converter = Bru2()
    >>> converter.inputs.input_dir = "brukerdir"
    >>> converter.cmdline  # doctest: +ELLIPSIS
    'Bru2 -o .../data/brukerdir brukerdir'
    """
    input_spec = Bru2InputSpec
    output_spec = Bru2OutputSpec
    _cmd = 'Bru2'

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.output_filename):
            output_filename1 = os.path.abspath(self.inputs.output_filename)
        else:
            output_filename1 = self._gen_filename('output_filename')
        if self.inputs.compress:
            outputs['nii_file'] = output_filename1 + '.nii.gz'
        else:
            outputs['nii_file'] = output_filename1 + '.nii'
        return outputs

    def _gen_filename(self, name):
        if name == 'output_filename':
            outfile = os.path.join(os.getcwd(), os.path.basename(os.path.normpath(self.inputs.input_dir)))
            return outfile