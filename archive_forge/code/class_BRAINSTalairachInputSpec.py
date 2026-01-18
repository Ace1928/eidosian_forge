import os
from ...base import (
class BRAINSTalairachInputSpec(CommandLineInputSpec):
    AC = InputMultiPath(traits.Float, desc='Location of AC Point ', sep=',', argstr='--AC %s')
    ACisIndex = traits.Bool(desc='AC Point is Index', argstr='--ACisIndex ')
    PC = InputMultiPath(traits.Float, desc='Location of PC Point ', sep=',', argstr='--PC %s')
    PCisIndex = traits.Bool(desc='PC Point is Index', argstr='--PCisIndex ')
    SLA = InputMultiPath(traits.Float, desc='Location of SLA Point ', sep=',', argstr='--SLA %s')
    SLAisIndex = traits.Bool(desc='SLA Point is Index', argstr='--SLAisIndex ')
    IRP = InputMultiPath(traits.Float, desc='Location of IRP Point ', sep=',', argstr='--IRP %s')
    IRPisIndex = traits.Bool(desc='IRP Point is Index', argstr='--IRPisIndex ')
    inputVolume = File(desc='Input image used to define physical space of images', exists=True, argstr='--inputVolume %s')
    outputBox = traits.Either(traits.Bool, File(), hash_files=False, desc='Name of the resulting Talairach Bounding Box file', argstr='--outputBox %s')
    outputGrid = traits.Either(traits.Bool, File(), hash_files=False, desc='Name of the resulting Talairach Grid file', argstr='--outputGrid %s')