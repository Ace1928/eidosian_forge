import os
from glob import glob
from ...external.due import BibTeX
from ...utils.filemanip import split_filename, copyfile, which, fname_presuffix
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from ..mixins import CopyHeaderInterface
from .base import ANTSCommand, ANTSCommandInputSpec
class LaplacianThicknessInputSpec(ANTSCommandInputSpec):
    input_wm = File(argstr='%s', mandatory=True, copyfile=True, desc='white matter segmentation image', position=1)
    input_gm = File(argstr='%s', mandatory=True, copyfile=True, desc='gray matter segmentation image', position=2)
    output_image = traits.Str(desc='name of output file', argstr='%s', position=3, name_source=['input_wm'], name_template='%s_thickness', keep_extension=True, hash_files=False)
    smooth_param = traits.Float(argstr='%s', desc='Sigma of the Laplacian Recursive Image Filter (defaults to 1)', position=4)
    prior_thickness = traits.Float(argstr='%s', desc='Prior thickness (defaults to 500)', requires=['smooth_param'], position=5)
    dT = traits.Float(argstr='%s', desc='Time delta used during integration (defaults to 0.01)', requires=['prior_thickness'], position=6)
    sulcus_prior = traits.Float(argstr='%s', desc='Positive floating point number for sulcus prior. Authors said that 0.15 might be a reasonable value', requires=['dT'], position=7)
    tolerance = traits.Float(argstr='%s', desc='Tolerance to reach during optimization (defaults to 0.001)', requires=['sulcus_prior'], position=8)