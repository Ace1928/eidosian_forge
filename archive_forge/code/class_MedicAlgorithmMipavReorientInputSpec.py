import os
from ..base import (
class MedicAlgorithmMipavReorientInputSpec(CommandLineInputSpec):
    inSource = InputMultiPath(File, desc='Source', sep=';', argstr='--inSource %s')
    inTemplate = File(desc='Template', exists=True, argstr='--inTemplate %s')
    inNew = traits.Enum('Dicom axial', 'Dicom coronal', 'Dicom sagittal', 'User defined', desc='New image orientation', argstr='--inNew %s')
    inUser = traits.Enum('Unknown', 'Patient Right to Left', 'Patient Left to Right', 'Patient Posterior to Anterior', 'Patient Anterior to Posterior', 'Patient Inferior to Superior', 'Patient Superior to Inferior', desc='User defined X-axis orientation (image left to right)', argstr='--inUser %s')
    inUser2 = traits.Enum('Unknown', 'Patient Right to Left', 'Patient Left to Right', 'Patient Posterior to Anterior', 'Patient Anterior to Posterior', 'Patient Inferior to Superior', 'Patient Superior to Inferior', desc='User defined Y-axis orientation (image top to bottom)', argstr='--inUser2 %s')
    inUser3 = traits.Enum('Unknown', 'Patient Right to Left', 'Patient Left to Right', 'Patient Posterior to Anterior', 'Patient Anterior to Posterior', 'Patient Inferior to Superior', 'Patient Superior to Inferior', desc='User defined Z-axis orientation (into the screen)', argstr='--inUser3 %s')
    inUser4 = traits.Enum('Axial', 'Coronal', 'Sagittal', 'Unknown', desc='User defined Image Orientation', argstr='--inUser4 %s')
    inInterpolation = traits.Enum('Nearest Neighbor', 'Trilinear', 'Bspline 3rd order', 'Bspline 4th order', 'Cubic Lagrangian', 'Quintic Lagrangian', 'Heptic Lagrangian', 'Windowed Sinc', desc='Interpolation', argstr='--inInterpolation %s')
    inResolution = traits.Enum('Unchanged', 'Finest cubic', 'Coarsest cubic', 'Same as template', desc='Resolution', argstr='--inResolution %s')
    xPrefExt = traits.Enum('nrrd', desc='Output File Type', argstr='--xPrefExt %s')
    outReoriented = InputMultiPath(File, desc='Reoriented Volume', sep=';', argstr='--outReoriented %s')
    null = traits.Str(desc='Execution Time', argstr='--null %s')
    xDefaultMem = traits.Int(desc='Set default maximum heap size', argstr='-xDefaultMem %d')
    xMaxProcess = traits.Int(1, desc='Set default maximum number of processes.', argstr='-xMaxProcess %d', usedefault=True)