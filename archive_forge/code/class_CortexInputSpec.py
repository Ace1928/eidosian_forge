import os
import re as regex
from ..base import (
class CortexInputSpec(CommandLineInputSpec):
    inputHemisphereLabelFile = File(mandatory=True, desc='hemisphere / lobe label volume', argstr='-h %s')
    outputCerebrumMask = File(desc='output structure mask. If unspecified, output file name will be auto generated.', argstr='-o %s', genfile=True)
    inputTissueFractionFile = File(mandatory=True, desc='tissue fraction file (32-bit float)', argstr='-f %s')
    tissueFractionThreshold = traits.Float(50.0, usedefault=True, desc='tissue fraction threshold (percentage)', argstr='-p %f')
    computeWGBoundary = traits.Bool(True, usedefault=True, desc='compute WM/GM boundary', argstr='-w')
    computeGCBoundary = traits.Bool(desc='compute GM/CSF boundary', argstr='-g')
    includeAllSubcorticalAreas = traits.Bool(True, usedefault=True, desc='include all subcortical areas in WM mask', argstr='-a')
    verbosity = traits.Int(desc='verbosity level', argstr='-v %d')
    timer = traits.Bool(desc='timing function', argstr='--timer')