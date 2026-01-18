from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class AffScalarVolInputSpec(CommandLineInputSpec):
    in_file = File(desc='moving scalar volume', exists=True, argstr='-in %s', mandatory=True)
    out_file = File(desc='output filename', argstr='-out %s', name_source='in_file', name_template='%s_affxfmd', keep_extension=True)
    transform = File(exists=True, argstr='-trans %s', xor=['target', 'translation', 'euler', 'deformation'], desc='transform to apply: specify an input transformation file; parameters input will be ignored')
    interpolation = traits.Enum('trilinear', 'NN', usedefault=True, argstr='-interp %s', desc='trilinear or nearest neighbor interpolation')
    target = File(exists=True, argstr='-target %s', xor=['transform'], desc='output volume specification read from the target volume if specified')
    translation = traits.Tuple((traits.Float(), traits.Float(), traits.Float()), desc='translation (x,y,z) in mm', argstr='-translation %g %g %g', xor=['transform'])
    euler = traits.Tuple((traits.Float(), traits.Float(), traits.Float()), desc='(theta, phi, psi) in degrees', xor=['transform'], argstr='-euler %g %g %g')
    deformation = traits.Tuple((traits.Float(),) * 6, desc='(xx,yy,zz,xy,yz,xz)', xor=['transform'], argstr='-deformation %g %g %g %g %g %g')