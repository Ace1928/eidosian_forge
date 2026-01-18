import os
import re as regex
from ..base import (
class PialmeshInputSpec(CommandLineInputSpec):
    inputSurfaceFile = File(mandatory=True, desc='input file', argstr='-i %s')
    outputSurfaceFile = File(desc='output file. If unspecified, output file name will be auto generated.', argstr='-o %s', genfile=True)
    verbosity = traits.Int(desc='verbosity', argstr='-v %d')
    inputTissueFractionFile = File(mandatory=True, desc='floating point (32) tissue fraction image', argstr='-f %s')
    numIterations = traits.Int(100, usedefault=True, desc='number of iterations', argstr='-n %d')
    searchRadius = traits.Float(1, usedefault=True, desc='search radius', argstr='-r %f')
    stepSize = traits.Float(0.4, usedefault=True, desc='step size', argstr='-s %f')
    inputMaskFile = File(mandatory=True, desc='restrict growth to mask file region', argstr='-m %s')
    maxThickness = traits.Float(20, usedefault=True, desc='maximum allowed tissue thickness', argstr='--max %f')
    tissueThreshold = traits.Float(1.05, usedefault=True, desc='tissue threshold', argstr='-t %f')
    outputInterval = traits.Int(10, usedefault=True, desc='output interval', argstr='--interval %d')
    exportPrefix = traits.Str(desc='prefix for exporting surfaces if interval is set', argstr='--prefix %s')
    laplacianSmoothing = traits.Float(0.025, usedefault=True, desc='apply Laplacian smoothing', argstr='--smooth %f')
    timer = traits.Bool(desc='show timing', argstr='--timer')
    recomputeNormals = traits.Bool(desc='recompute normals at each iteration', argstr='--norm')
    normalSmoother = traits.Float(0.2, usedefault=True, desc='strength of normal smoother.', argstr='--nc %f')
    tangentSmoother = traits.Float(desc='strength of tangential smoother.', argstr='--tc %f')