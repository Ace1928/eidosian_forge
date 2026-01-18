import numpy as np
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class StatsInput(CommandLineInputSpec):
    """Input Spec for seg_stats interfaces."""
    in_file = File(position=2, argstr='%s', exists=True, mandatory=True, desc='image to operate on')
    mask_file = File(exists=True, position=-2, argstr='-m %s', desc='statistics within the masked area')
    desc = 'Only estimate statistics if voxel is larger than <float>'
    larger_voxel = traits.Float(argstr='-t %f', position=-3, desc=desc)