import os
from ...base import (
class fcsv_to_hdf5OutputSpec(TraitedSpec):
    landmarksInformationFile = File(desc=',         name of HDF5 file to write matrices into,       ', exists=True)
    modelFile = File(desc=',         name of HDF5 file containing BRAINSConstellationDetector Model file (LLSMatrices, LLSMeans and LLSSearchRadii),       ', exists=True)