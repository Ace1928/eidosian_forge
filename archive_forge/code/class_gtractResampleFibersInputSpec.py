import os
from ...base import (
class gtractResampleFibersInputSpec(CommandLineInputSpec):
    inputForwardDeformationFieldVolume = File(desc='Required: input forward deformation field image file name', exists=True, argstr='--inputForwardDeformationFieldVolume %s')
    inputReverseDeformationFieldVolume = File(desc='Required: input reverse deformation field image file name', exists=True, argstr='--inputReverseDeformationFieldVolume %s')
    inputTract = File(desc='Required: name of input vtkPolydata file containing tract lines.', exists=True, argstr='--inputTract %s')
    outputTract = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: name of output vtkPolydata file containing tract lines and the point data collected along them.', argstr='--outputTract %s')
    writeXMLPolyDataFile = traits.Bool(desc='Flag to make use of the XML format for vtkPolyData fiber tracts.', argstr='--writeXMLPolyDataFile ')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')