from .base import FSLCommand, FSLCommandInputSpec
from ..base import TraitedSpec, File, traits
class B0CalcInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='-i %s', position=0, desc='filename of input image (usually a tissue/air segmentation)')
    out_file = File(argstr='-o %s', position=1, name_source=['in_file'], name_template='%s_b0field', output_name='out_file', desc='filename of B0 output volume')
    x_grad = traits.Float(0.0, usedefault=True, argstr='--gx=%0.4f', desc='Value for zeroth-order x-gradient field (per mm)')
    y_grad = traits.Float(0.0, usedefault=True, argstr='--gy=%0.4f', desc='Value for zeroth-order y-gradient field (per mm)')
    z_grad = traits.Float(0.0, usedefault=True, argstr='--gz=%0.4f', desc='Value for zeroth-order z-gradient field (per mm)')
    x_b0 = traits.Float(0.0, usedefault=True, argstr='--b0x=%0.2f', xor=['xyz_b0'], desc='Value for zeroth-order b0 field (x-component), in Tesla')
    y_b0 = traits.Float(0.0, usedefault=True, argstr='--b0y=%0.2f', xor=['xyz_b0'], desc='Value for zeroth-order b0 field (y-component), in Tesla')
    z_b0 = traits.Float(1.0, usedefault=True, argstr='--b0=%0.2f', xor=['xyz_b0'], desc='Value for zeroth-order b0 field (z-component), in Tesla')
    xyz_b0 = traits.Tuple(traits.Float, traits.Float, traits.Float, argstr='--b0x=%0.2f --b0y=%0.2f --b0=%0.2f', xor=['x_b0', 'y_b0', 'z_b0'], desc='Zeroth-order B0 field in Tesla')
    delta = traits.Float(-9.45e-06, usedefault=True, argstr='-d %e', desc='Delta value (chi_tissue - chi_air)')
    chi_air = traits.Float(4e-07, usedefault=True, argstr='--chi0=%e', desc='susceptibility of air')
    compute_xyz = traits.Bool(False, usedefault=True, argstr='--xyz', desc='calculate and save all 3 field components (i.e. x,y,z)')
    extendboundary = traits.Float(1.0, usedefault=True, argstr='--extendboundary=%0.2f', desc='Relative proportion to extend voxels at boundary')
    directconv = traits.Bool(False, usedefault=True, argstr='--directconv', desc='use direct (image space) convolution, not FFT')