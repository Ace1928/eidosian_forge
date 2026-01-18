import os
import re as regex
from ..base import (
class ScrubmaskInputSpec(CommandLineInputSpec):
    inputMaskFile = File(mandatory=True, desc='input structure mask file', argstr='-i %s')
    outputMaskFile = File(desc='output structure mask file. If unspecified, output file name will be auto generated.', argstr='-o %s', genfile=True)
    backgroundFillThreshold = traits.Int(2, usedefault=True, desc='background fill threshold', argstr='-b %d')
    foregroundTrimThreshold = traits.Int(0, usedefault=True, desc='foreground trim threshold', argstr='-f %d')
    numberIterations = traits.Int(desc='number of iterations', argstr='-n %d')
    verbosity = traits.Int(desc='verbosity (0=silent)', argstr='-v %d')
    timer = traits.Bool(desc='timing function', argstr='--timer')