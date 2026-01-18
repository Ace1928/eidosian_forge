import os
from ...base import (
class gtractFastMarchingTrackingInputSpec(CommandLineInputSpec):
    inputTensorVolume = File(desc='Required: input tensor image file name', exists=True, argstr='--inputTensorVolume %s')
    inputAnisotropyVolume = File(desc='Required: input anisotropy image file name', exists=True, argstr='--inputAnisotropyVolume %s')
    inputCostVolume = File(desc='Required: input vcl_cost image file name', exists=True, argstr='--inputCostVolume %s')
    inputStartingSeedsLabelMapVolume = File(desc='Required: input starting seeds LabelMap image file name', exists=True, argstr='--inputStartingSeedsLabelMapVolume %s')
    startingSeedsLabel = traits.Int(desc='Label value for Starting Seeds', argstr='--startingSeedsLabel %d')
    outputTract = traits.Either(traits.Bool, File(), hash_files=False, desc='Required: name of output vtkPolydata file containing tract lines and the point data collected along them.', argstr='--outputTract %s')
    writeXMLPolyDataFile = traits.Bool(desc='Flag to make use of the XML format for vtkPolyData fiber tracts.', argstr='--writeXMLPolyDataFile ')
    numberOfIterations = traits.Int(desc='Number of iterations used for the optimization', argstr='--numberOfIterations %d')
    seedThreshold = traits.Float(desc='Anisotropy threshold used for seed selection', argstr='--seedThreshold %f')
    trackingThreshold = traits.Float(desc='Anisotropy threshold used for fiber tracking', argstr='--trackingThreshold %f')
    costStepSize = traits.Float(desc='Cost image sub-voxel sampling', argstr='--costStepSize %f')
    maximumStepSize = traits.Float(desc='Maximum step size to move when tracking', argstr='--maximumStepSize %f')
    minimumStepSize = traits.Float(desc='Minimum step size to move when tracking', argstr='--minimumStepSize %f')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')