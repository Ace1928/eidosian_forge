import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class ImageMathInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(3, usedefault=True, position=1, argstr='%d', desc='dimension of output image')
    output_image = File(position=2, argstr='%s', name_source=['op1'], name_template='%s_maths', desc='output image file', keep_extension=True)
    operation = traits.Enum('m', 'vm', '+', 'v+', '-', 'v-', '/', '^', 'max', 'exp', 'addtozero', 'overadd', 'abs', 'total', 'mean', 'vtotal', 'Decision', 'Neg', 'Project', 'G', 'MD', 'ME', 'MO', 'MC', 'GD', 'GE', 'GO', 'GC', 'ExtractContours', 'Translate', '4DTensorTo3DTensor', 'ExtractVectorComponent', 'TensorColor', 'TensorFA', 'TensorFADenominator', 'TensorFANumerator', 'TensorMeanDiffusion', 'TensorRadialDiffusion', 'TensorAxialDiffusion', 'TensorEigenvalue', 'TensorToVector', 'TensorToVectorComponent', 'TensorMask', 'Byte', 'CorruptImage', 'D', 'MaurerDistance', 'ExtractSlice', 'FillHoles', 'Convolve', 'Finite', 'FlattenImage', 'GetLargestComponent', 'Grad', 'RescaleImage', 'WindowImage', 'NeighborhoodStats', 'ReplicateDisplacement', 'ReplicateImage', 'LabelStats', 'Laplacian', 'Canny', 'Lipschitz', 'MTR', 'Normalize', 'PadImage', 'SigmoidImage', 'Sharpen', 'UnsharpMask', 'PValueImage', 'ReplaceVoxelValue', 'SetTimeSpacing', 'SetTimeSpacingWarp', 'stack', 'ThresholdAtMean', 'TriPlanarView', 'TruncateImageIntensity', mandatory=True, position=3, argstr='%s', desc='mathematical operations')
    op1 = File(exists=True, mandatory=True, position=-3, argstr='%s', desc='first operator')
    op2 = traits.Either(File(exists=True), Str, position=-2, argstr='%s', desc='second operator')
    args = Str(position=-1, argstr='%s', desc='Additional parameters to the command')
    copy_header = traits.Bool(True, usedefault=True, desc='copy headers of the original image into the output (corrected) file')