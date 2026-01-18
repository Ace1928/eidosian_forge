import os
import re as regex
from ..base import (
class HemisplitInputSpec(CommandLineInputSpec):
    inputSurfaceFile = File(mandatory=True, desc='input surface', argstr='-i %s')
    inputHemisphereLabelFile = File(mandatory=True, desc='input hemisphere label volume', argstr='-l %s')
    outputLeftHemisphere = File(desc='output surface file, left hemisphere. If unspecified, output file name will be auto generated.', argstr='--left %s', genfile=True)
    outputRightHemisphere = File(desc='output surface file, right hemisphere. If unspecified, output file name will be auto generated.', argstr='--right %s', genfile=True)
    pialSurfaceFile = File(desc='pial surface file -- must have same geometry as input surface', argstr='-p %s')
    outputLeftPialHemisphere = File(desc='output pial surface file, left hemisphere. If unspecified, output file name will be auto generated.', argstr='-pl %s', genfile=True)
    outputRightPialHemisphere = File(desc='output pial surface file, right hemisphere. If unspecified, output file name will be auto generated.', argstr='-pr %s', genfile=True)
    verbosity = traits.Int(desc='verbosity (0 = silent)', argstr='-v %d')
    timer = traits.Bool(desc='timing function', argstr='--timer')