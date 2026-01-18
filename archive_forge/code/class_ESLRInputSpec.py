import os
from ...base import (
class ESLRInputSpec(CommandLineInputSpec):
    inputVolume = File(desc='Input Label Volume', exists=True, argstr='--inputVolume %s')
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc='Output Label Volume', argstr='--outputVolume %s')
    low = traits.Int(desc='The lower bound of the labels to be used.', argstr='--low %d')
    high = traits.Int(desc='The higher bound of the labels to be used.', argstr='--high %d')
    closingSize = traits.Int(desc='The closing size for hole filling.', argstr='--closingSize %d')
    openingSize = traits.Int(desc='The opening size for hole filling.', argstr='--openingSize %d')
    safetySize = traits.Int(desc='The safetySize size for the clipping region.', argstr='--safetySize %d')
    preserveOutside = traits.Bool(desc='For values outside the specified range, preserve those values.', argstr='--preserveOutside ')
    numberOfThreads = traits.Int(desc='Explicitly specify the maximum number of threads to use.', argstr='--numberOfThreads %d')