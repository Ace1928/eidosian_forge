import os
from ...base import (
class BRAINSLinearModelerEPCAInputSpec(CommandLineInputSpec):
    inputTrainingList = File(desc='Input Training Landmark List Filename,             ', exists=True, argstr='--inputTrainingList %s')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')