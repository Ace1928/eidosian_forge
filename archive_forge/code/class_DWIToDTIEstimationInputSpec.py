from nipype.interfaces.base import (
import os
class DWIToDTIEstimationInputSpec(CommandLineInputSpec):
    inputVolume = File(position=-3, desc='Input DWI volume', exists=True, argstr='%s')
    mask = File(desc='Mask where the tensors will be computed', exists=True, argstr='--mask %s')
    outputTensor = traits.Either(traits.Bool, File(), position=-2, hash_files=False, desc='Estimated DTI volume', argstr='%s')
    outputBaseline = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Estimated baseline volume', argstr='%s')
    enumeration = traits.Enum('LS', 'WLS', desc='LS: Least Squares, WLS: Weighted Least Squares', argstr='--enumeration %s')
    shiftNeg = traits.Bool(desc='Shift eigenvalues so all are positive (accounts for bad tensors related to noise or acquisition error)', argstr='--shiftNeg ')