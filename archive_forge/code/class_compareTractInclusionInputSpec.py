import os
from ...base import (
class compareTractInclusionInputSpec(CommandLineInputSpec):
    testFiber = File(desc='Required: test fiber tract file name', exists=True, argstr='--testFiber %s')
    standardFiber = File(desc='Required: standard fiber tract file name', exists=True, argstr='--standardFiber %s')
    closeness = traits.Float(desc='Closeness of every test fiber to some fiber in the standard tract, computed as a sum of squares of spatial differences of standard points', argstr='--closeness %f')
    numberOfPoints = traits.Int(desc='Number of points in comparison fiber pairs', argstr='--numberOfPoints %d')
    testForBijection = traits.Bool(desc='Flag to apply the closeness criterion both ways', argstr='--testForBijection ')
    testForFiberCardinality = traits.Bool(desc='Flag to require the same number of fibers in both tracts', argstr='--testForFiberCardinality ')
    writeXMLPolyDataFile = traits.Bool(desc='Flag to make use of XML files when reading and writing vtkPolyData.', argstr='--writeXMLPolyDataFile ')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')