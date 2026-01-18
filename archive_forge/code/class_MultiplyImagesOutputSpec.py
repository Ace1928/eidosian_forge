import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class MultiplyImagesOutputSpec(TraitedSpec):
    output_product_image = File(exists=True, desc='average image file')