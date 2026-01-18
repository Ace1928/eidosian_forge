import os
import re as regex
from ..base import (
class DewispInputSpec(CommandLineInputSpec):
    inputMaskFile = File(mandatory=True, desc='input file', argstr='-i %s')
    outputMaskFile = File(desc='output file. If unspecified, output file name will be auto generated.', argstr='-o %s', genfile=True)
    verbosity = traits.Int(desc='verbosity', argstr='-v %d')
    sizeThreshold = traits.Int(desc='size threshold', argstr='-t %d')
    maximumIterations = traits.Int(desc='maximum number of iterations', argstr='-n %d')
    timer = traits.Bool(desc='time processing', argstr='--timer')