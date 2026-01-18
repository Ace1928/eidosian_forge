from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class DiffeoScalarVol(CommandLineDtitk):
    """
    Applies diffeomorphic transform to a scalar volume

    Example
    -------

    >>> from nipype.interfaces import dtitk
    >>> node = dtitk.DiffeoScalarVol()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.transform = 'im_warp.df.nii'
    >>> node.cmdline
    'deformationScalarVolume -in im1.nii -interp 0 -out im1_diffeoxfmd.nii
     -trans im_warp.df.nii'
    >>> node.run() # doctest: +SKIP
    """
    input_spec = DiffeoScalarVolInputSpec
    output_spec = DiffeoScalarVolOutputSpec
    _cmd = 'deformationScalarVolume'

    def _format_arg(self, name, spec, value):
        if name == 'resampling_type':
            value = {'forward': 0, 'backward': 1}[value]
        elif name == 'interpolation':
            value = {'trilinear': 0, 'NN': 1}[value]
        return super(DiffeoScalarVol, self)._format_arg(name, spec, value)