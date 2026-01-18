from nipype.interfaces.base import (
import os
class DiffusionTensorScalarMeasurementsInputSpec(CommandLineInputSpec):
    inputVolume = File(position=-3, desc='Input DTI volume', exists=True, argstr='%s')
    outputScalar = traits.Either(traits.Bool, File(), position=-1, hash_files=False, desc='Scalar volume derived from tensor', argstr='%s')
    enumeration = traits.Enum('Trace', 'Determinant', 'RelativeAnisotropy', 'FractionalAnisotropy', 'Mode', 'LinearMeasure', 'PlanarMeasure', 'SphericalMeasure', 'MinEigenvalue', 'MidEigenvalue', 'MaxEigenvalue', 'MaxEigenvalueProjectionX', 'MaxEigenvalueProjectionY', 'MaxEigenvalueProjectionZ', 'RAIMaxEigenvecX', 'RAIMaxEigenvecY', 'RAIMaxEigenvecZ', 'MaxEigenvecX', 'MaxEigenvecY', 'MaxEigenvecZ', 'D11', 'D22', 'D33', 'ParallelDiffusivity', 'PerpendicularDffusivity', desc='An enumeration of strings', argstr='--enumeration %s')