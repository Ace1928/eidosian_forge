from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class Diffeo(CommandLineDtitk):
    """Performs diffeomorphic registration between two tensor volumes

    Example
    -------

    >>> from nipype.interfaces import dtitk
    >>> node = dtitk.Diffeo()
    >>> node.inputs.fixed_file = 'im1.nii'
    >>> node.inputs.moving_file = 'im2.nii'
    >>> node.inputs.mask_file = 'mask.nii'
    >>> node.inputs.legacy = 1
    >>> node.inputs.n_iters = 6
    >>> node.inputs.ftol = 0.002
    >>> node.cmdline
    'dti_diffeomorphic_reg im1.nii im2.nii mask.nii 1 6 0.002'
    >>> node.run() # doctest: +SKIP
    """
    input_spec = DiffeoInputSpec
    output_spec = DiffeoOutputSpec
    _cmd = 'dti_diffeomorphic_reg'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        moving = self.inputs.moving_file
        outputs['out_file_xfm'] = fname_presuffix(moving, suffix='_diffeo.df')
        outputs['out_file'] = fname_presuffix(moving, suffix='_diffeo')
        return outputs