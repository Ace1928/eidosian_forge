import os
from .base import (
from ..utils.filemanip import fname_presuffix
from ..external.due import BibTeX
class PETPVCInputSpec(CommandLineInputSpec):
    in_file = File(desc='PET image file', exists=True, mandatory=True, argstr='-i %s')
    out_file = File(desc='Output file', genfile=True, hash_files=False, argstr='-o %s')
    mask_file = File(desc='Mask image file', exists=True, mandatory=True, argstr='-m %s')
    pvc = traits.Enum(pvc_methods, mandatory=True, argstr='-p %s', desc='Desired PVC method:\n\n    * Geometric transfer matrix -- ``GTM``\n    * Labbe approach -- ``LABBE``\n    * Richardson-Lucy -- ``RL``\n    * Van-Cittert -- ``VC``\n    * Region-based voxel-wise correction -- ``RBV``\n    * RBV with Labbe -- ``LABBE+RBV``\n    * RBV with Van-Cittert -- ``RBV+VC``\n    * RBV with Richardson-Lucy -- ``RBV+RL``\n    * RBV with Labbe and Van-Cittert -- ``LABBE+RBV+VC``\n    * RBV with Labbe and Richardson-Lucy -- ``LABBE+RBV+RL``\n    * Multi-target correction -- ``MTC``\n    * MTC with Labbe -- ``LABBE+MTC``\n    * MTC with Van-Cittert -- ``MTC+VC``\n    * MTC with Richardson-Lucy -- ``MTC+RL``\n    * MTC with Labbe and Van-Cittert -- ``LABBE+MTC+VC``\n    * MTC with Labbe and Richardson-Lucy -- ``LABBE+MTC+RL``\n    * Iterative Yang -- ``IY``\n    * Iterative Yang with Van-Cittert -- ``IY+VC``\n    * Iterative Yang with Richardson-Lucy -- ``IY+RL``\n    * Muller Gartner -- ``MG``\n    * Muller Gartner with Van-Cittert -- ``MG+VC``\n    * Muller Gartner with Richardson-Lucy -- ``MG+RL``\n\n')
    fwhm_x = traits.Float(desc='The full-width at half maximum in mm along x-axis', mandatory=True, argstr='-x %.4f')
    fwhm_y = traits.Float(desc='The full-width at half maximum in mm along y-axis', mandatory=True, argstr='-y %.4f')
    fwhm_z = traits.Float(desc='The full-width at half maximum in mm along z-axis', mandatory=True, argstr='-z %.4f')
    debug = traits.Bool(desc='Prints debug information', usedefault=True, default_value=False, argstr='-d')
    n_iter = traits.Int(desc='Number of iterations', default_value=10, usedefault=True, argstr='-n %d')
    n_deconv = traits.Int(desc='Number of deconvolution iterations', default_value=10, usedefault=True, argstr='-k %d')
    alpha = traits.Float(desc='Alpha value', default_value=1.5, usedefault=True, argstr='-a %.4f')
    stop_crit = traits.Float(desc='Stopping criterion', default_value=0.01, usedefault=True, argstr='-s %.4f')