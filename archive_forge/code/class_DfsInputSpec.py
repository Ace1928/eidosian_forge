import os
import re as regex
from ..base import (
class DfsInputSpec(CommandLineInputSpec):
    inputVolumeFile = File(mandatory=True, desc='input 3D volume', argstr='-i %s')
    outputSurfaceFile = File(desc='output surface mesh file. If unspecified, output file name will be auto generated.', argstr='-o %s', genfile=True)
    inputShadingVolume = File(desc='shade surface model with data from image volume', argstr='-c %s')
    smoothingIterations = traits.Int(10, usedefault=True, desc='number of smoothing iterations', argstr='-n %d')
    smoothingConstant = traits.Float(0.5, usedefault=True, desc='smoothing constant', argstr='-a %f')
    curvatureWeighting = traits.Float(5.0, usedefault=True, desc='curvature weighting', argstr='-w %f')
    scalingPercentile = traits.Float(desc='scaling percentile', argstr='-f %f')
    nonZeroTessellation = traits.Bool(desc='tessellate non-zero voxels', argstr='-nz', xor=('nonZeroTessellation', 'specialTessellation'))
    tessellationThreshold = traits.Float(desc='To be used with specialTessellation. Set this value first, then set specialTessellation value.\nUsage: tessellate voxels greater_than, less_than, or equal_to <tessellationThreshold>', argstr='%f')
    specialTessellation = traits.Enum('greater_than', 'less_than', 'equal_to', desc='To avoid throwing a UserWarning, set tessellationThreshold first. Then set this attribute.\nUsage: tessellate voxels greater_than, less_than, or equal_to <tessellationThreshold>', argstr='%s', xor=('nonZeroTessellation', 'specialTessellation'), requires=['tessellationThreshold'], position=-1)
    zeroPadFlag = traits.Bool(desc='zero-pad volume (avoids clipping at edges)', argstr='-z')
    noNormalsFlag = traits.Bool(desc='do not compute vertex normals', argstr='--nonormals')
    postSmoothFlag = traits.Bool(desc='smooth vertices after coloring', argstr='--postsmooth')
    verbosity = traits.Int(desc='verbosity (0 = quiet)', argstr='-v %d')
    timer = traits.Bool(desc='timing function', argstr='--timer')