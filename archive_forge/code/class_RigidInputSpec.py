from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class RigidInputSpec(CommandLineInputSpec):
    fixed_file = File(desc='fixed tensor volume', exists=True, mandatory=True, position=0, argstr='%s', copyfile=False)
    moving_file = File(desc='moving tensor volume', exists=True, mandatory=True, position=1, argstr='%s', copyfile=False)
    similarity_metric = traits.Enum('EDS', 'GDS', 'DDS', 'NMI', mandatory=True, position=2, argstr='%s', desc='similarity metric', usedefault=True)
    sampling_xyz = traits.Tuple((4, 4, 4), mandatory=True, position=3, argstr='%g %g %g', usedefault=True, desc='dist between samp points (mm) (x,y,z)')
    ftol = traits.Float(mandatory=True, position=4, argstr='%g', desc='cost function tolerance', default_value=0.01, usedefault=True)
    initialize_xfm = File(copyfile=True, desc='Initialize w/DTITK-FORMATaffine', position=5, argstr='%s', exists=True)