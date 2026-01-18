import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class AffineInitializerInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(3, 2, usedefault=True, position=0, argstr='%s', desc='dimension')
    fixed_image = File(exists=True, mandatory=True, position=1, argstr='%s', desc='reference image')
    moving_image = File(exists=True, mandatory=True, position=2, argstr='%s', desc='moving image')
    out_file = File('transform.mat', usedefault=True, position=3, argstr='%s', desc='output transform file')
    search_factor = traits.Float(15.0, usedefault=True, position=4, argstr='%f', desc='increments (degrees) for affine search')
    radian_fraction = traits.Range(0.0, 1.0, value=0.1, usedefault=True, position=5, argstr='%f', desc='search this arc +/- principal axes')
    principal_axes = traits.Bool(False, usedefault=True, position=6, argstr='%d', desc='whether the rotation is searched around an initial principal axis alignment.')
    local_search = traits.Int(10, usedefault=True, position=7, argstr='%d', desc=' determines if a local optimization is run at each search point for the set number of iterations')