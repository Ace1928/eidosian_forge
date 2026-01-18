from ..base import TraitedSpec, CommandLineInputSpec, File, traits, isdefined
from ...utils.filemanip import fname_presuffix
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class TVtool(CommandLineDtitk):
    """
    Calculates a tensor metric volume from a tensor volume.

    Example
    -------
    >>> from nipype.interfaces import dtitk
    >>> node = dtitk.TVtool()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.in_flag = 'fa'
    >>> node.cmdline
    'TVtool -in im1.nii -fa -out im1_fa.nii'
    >>> node.run() # doctest: +SKIP

    """
    input_spec = TVtoolInputSpec
    output_spec = TVtoolOutputSpec
    _cmd = 'TVtool'

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_file = self.inputs.out_file
        if not isdefined(out_file):
            out_file = self._gen_filename('out_file')
        outputs['out_file'] = os.path.abspath(out_file)
        return outputs

    def _gen_filename(self, name):
        if name != 'out_file':
            return
        return fname_presuffix(os.path.basename(self.inputs.in_file), suffix='_' + self.inputs.in_flag)