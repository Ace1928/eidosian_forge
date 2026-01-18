import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class FSL2SchemeInputSpec(StdOutCommandLineInputSpec):
    bvec_file = File(exists=True, argstr='-bvecfile %s', mandatory=True, position=1, desc='b vector file')
    bval_file = File(exists=True, argstr='-bvalfile %s', mandatory=True, position=2, desc='b value file')
    numscans = traits.Int(argstr='-numscans %d', units='NA', desc='Output all measurements numerous (n) times, used when combining multiple scans from the same imaging session.')
    interleave = traits.Bool(argstr='-interleave', desc='Interleave repeated scans. Only used with -numscans.')
    bscale = traits.Float(argstr='-bscale %d', units='NA', desc='Scaling factor to convert the b-values into different units. Default is 10^6.')
    diffusiontime = traits.Float(argstr='-diffusiontime %f', units='NA', desc='Diffusion time')
    flipx = traits.Bool(argstr='-flipx', desc='Negate the x component of all the vectors.')
    flipy = traits.Bool(argstr='-flipy', desc='Negate the y component of all the vectors.')
    flipz = traits.Bool(argstr='-flipz', desc='Negate the z component of all the vectors.')
    usegradmod = traits.Bool(argstr='-usegradmod', desc='Use the gradient magnitude to scale b. This option has no effect if your gradient directions have unit magnitude.')