import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class ThresholdImage(ANTSCommand, CopyHeaderInterface):
    """
    Apply thresholds on images.

    Examples
    --------
    >>> thres = ThresholdImage(dimension=3)
    >>> thres.inputs.input_image = 'structural.nii'
    >>> thres.inputs.output_image = 'output.nii.gz'
    >>> thres.inputs.th_low = 0.5
    >>> thres.inputs.th_high = 1.0
    >>> thres.inputs.inside_value = 1.0
    >>> thres.inputs.outside_value = 0.0
    >>> thres.cmdline  #doctest: +ELLIPSIS
    'ThresholdImage 3 structural.nii output.nii.gz 0.500000 1.000000 1.000000 0.000000'

    >>> thres = ThresholdImage(dimension=3)
    >>> thres.inputs.input_image = 'structural.nii'
    >>> thres.inputs.output_image = 'output.nii.gz'
    >>> thres.inputs.mode = 'Kmeans'
    >>> thres.inputs.num_thresholds = 4
    >>> thres.cmdline  #doctest: +ELLIPSIS
    'ThresholdImage 3 structural.nii output.nii.gz Kmeans 4'

    """
    _cmd = 'ThresholdImage'
    input_spec = ThresholdImageInputSpec
    output_spec = ThresholdImageOutputSpec
    _copy_header_map = {'output_image': 'input_image'}