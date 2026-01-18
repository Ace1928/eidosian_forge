from ..base import TraitedSpec, CommandLineInputSpec, File, traits, isdefined
from ...utils.filemanip import fname_presuffix
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class TVAdjustVoxSp(CommandLineDtitk):
    """
     Adjusts the voxel space of a tensor volume.

    Example
    -------
    >>> from nipype.interfaces import dtitk
    >>> node = dtitk.TVAdjustVoxSp()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.target_file = 'im2.nii'
    >>> node.cmdline
    'TVAdjustVoxelspace -in im1.nii -out im1_avs.nii -target im2.nii'
    >>> node.run() # doctest: +SKIP

    """
    input_spec = TVAdjustVoxSpInputSpec
    output_spec = TVAdjustVoxSpOutputSpec
    _cmd = 'TVAdjustVoxelspace'