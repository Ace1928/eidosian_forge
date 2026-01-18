from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class Rigid(CommandLineDtitk):
    """Performs rigid registration between two tensor volumes

    Example
    -------

    >>> from nipype.interfaces import dtitk
    >>> node = dtitk.Rigid()
    >>> node.inputs.fixed_file = 'im1.nii'
    >>> node.inputs.moving_file = 'im2.nii'
    >>> node.inputs.similarity_metric = 'EDS'
    >>> node.inputs.sampling_xyz = (4,4,4)
    >>> node.inputs.ftol = 0.01
    >>> node.cmdline
    'dti_rigid_reg im1.nii im2.nii EDS 4 4 4 0.01'
    >>> node.run() # doctest: +SKIP
    """
    input_spec = RigidInputSpec
    output_spec = RigidOutputSpec
    _cmd = 'dti_rigid_reg'
    "def _format_arg(self, name, spec, value):\n        if name == 'initialize_xfm':\n            value = 1\n        return super(Rigid, self)._format_arg(name, spec, value)"

    def _run_interface(self, runtime):
        runtime = super(Rigid, self)._run_interface(runtime)
        if ".aff doesn't exist or can't be opened" in runtime.stderr:
            self.raise_exception(runtime)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        moving = self.inputs.moving_file
        outputs['out_file_xfm'] = fname_presuffix(moving, suffix='.aff', use_ext=False)
        outputs['out_file'] = fname_presuffix(moving, suffix='_aff')
        return outputs