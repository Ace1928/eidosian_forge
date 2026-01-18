from ..base import TraitedSpec, CommandLineInputSpec, File, traits, isdefined
from ...utils.filemanip import fname_presuffix
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class SVAdjustVoxSp(CommandLineDtitk):
    """
    Adjusts the voxel space of a scalar volume.

    Example
    -------
    >>> from nipype.interfaces import dtitk
    >>> node = dtitk.SVAdjustVoxSp()
    >>> node.inputs.in_file = 'im1.nii'
    >>> node.inputs.target_file = 'im2.nii'
    >>> node.cmdline
    'SVAdjustVoxelspace -in im1.nii -out im1_avs.nii -target im2.nii'
    >>> node.run() # doctest: +SKIP

    """
    input_spec = SVAdjustVoxSpInputSpec
    output_spec = SVAdjustVoxSpOutputSpec
    _cmd = 'SVAdjustVoxelspace'