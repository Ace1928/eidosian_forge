from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class Affine(Rigid):
    """Performs affine registration between two tensor volumes

    Example
    -------

    >>> from nipype.interfaces import dtitk
    >>> node = dtitk.Affine()
    >>> node.inputs.fixed_file = 'im1.nii'
    >>> node.inputs.moving_file = 'im2.nii'
    >>> node.inputs.similarity_metric = 'EDS'
    >>> node.inputs.sampling_xyz = (4,4,4)
    >>> node.inputs.ftol = 0.01
    >>> node.inputs.initialize_xfm = 'im_affine.aff'
    >>> node.cmdline
    'dti_affine_reg im1.nii im2.nii EDS 4 4 4 0.01 im_affine.aff'
    >>> node.run() # doctest: +SKIP
    """
    _cmd = 'dti_affine_reg'