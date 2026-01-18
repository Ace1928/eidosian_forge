import os
import re as regex
from ..base import (
class BseInputSpec(CommandLineInputSpec):
    inputMRIFile = File(mandatory=True, argstr='-i %s', desc='input MRI volume')
    outputMRIVolume = File(desc='output brain-masked MRI volume. If unspecified, output file name will be auto generated.', argstr='-o %s', hash_files=False, genfile=True)
    outputMaskFile = File(desc='save smooth brain mask. If unspecified, output file name will be auto generated.', argstr='--mask %s', hash_files=False, genfile=True)
    diffusionConstant = traits.Float(25, usedefault=True, desc='diffusion constant', argstr='-d %f')
    diffusionIterations = traits.Int(3, usedefault=True, desc='diffusion iterations', argstr='-n %d')
    edgeDetectionConstant = traits.Float(0.64, usedefault=True, desc='edge detection constant', argstr='-s %f')
    radius = traits.Float(1, usedefault=True, desc='radius of erosion/dilation filter', argstr='-r %f')
    dilateFinalMask = traits.Bool(True, usedefault=True, desc='dilate final mask', argstr='-p')
    trim = traits.Bool(True, usedefault=True, desc='trim brainstem', argstr='--trim')
    outputDiffusionFilter = File(desc='diffusion filter output', argstr='--adf %s', hash_files=False)
    outputEdgeMap = File(desc='edge map output', argstr='--edge %s', hash_files=False)
    outputDetailedBrainMask = File(desc='save detailed brain mask', argstr='--hires %s', hash_files=False)
    outputCortexFile = File(desc='cortex file', argstr='--cortex %s', hash_files=False)
    verbosityLevel = traits.Float(1, usedefault=True, desc=' verbosity level (0=silent)', argstr='-v %f')
    noRotate = traits.Bool(desc='retain original orientation(default behavior will auto-rotate input NII files to LPI orientation)', argstr='--norotate')
    timer = traits.Bool(desc='show timing', argstr='--timer')