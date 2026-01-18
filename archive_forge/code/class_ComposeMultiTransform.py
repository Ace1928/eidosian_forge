import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class ComposeMultiTransform(ANTSCommand):
    """
    Take a set of transformations and convert them to a single transformation matrix/warpfield.

    Examples
    --------
    >>> from nipype.interfaces.ants import ComposeMultiTransform
    >>> compose_transform = ComposeMultiTransform()
    >>> compose_transform.inputs.dimension = 3
    >>> compose_transform.inputs.transforms = ['struct_to_template.mat', 'func_to_struct.mat']
    >>> compose_transform.cmdline
    'ComposeMultiTransform 3 struct_to_template_composed.mat
    struct_to_template.mat func_to_struct.mat'

    """
    _cmd = 'ComposeMultiTransform'
    input_spec = ComposeMultiTransformInputSpec
    output_spec = ComposeMultiTransformOutputSpec