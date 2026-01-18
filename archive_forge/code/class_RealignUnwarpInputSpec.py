import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class RealignUnwarpInputSpec(SPMCommandInputSpec):
    in_files = InputMultiObject(traits.Either(ImageFileSPM(exists=True), traits.List(ImageFileSPM(exists=True))), field='data.scans', mandatory=True, copyfile=True, desc='list of filenames to realign and unwarp')
    phase_map = File(field='data.pmscan', desc='Voxel displacement map to use in unwarping. Unlike SPM standard behaviour, the same map will be used for all sessions', copyfile=False)
    quality = traits.Range(low=0.0, high=1.0, field='eoptions.quality', desc='0.1 = fast, 1.0 = precise')
    fwhm = traits.Range(low=0.0, field='eoptions.fwhm', desc='gaussian smoothing kernel width')
    separation = traits.Range(low=0.0, field='eoptions.sep', desc='sampling separation in mm')
    register_to_mean = traits.Bool(field='eoptions.rtm', desc='Indicate whether realignment is done to the mean image')
    weight_img = File(exists=True, field='eoptions.weight', desc='filename of weighting image')
    interp = traits.Range(low=0, high=7, field='eoptions.einterp', desc='degree of b-spline used for interpolation')
    wrap = traits.List(traits.Int(), minlen=3, maxlen=3, field='eoptions.ewrap', desc='Check if interpolation should wrap in [x,y,z]')
    est_basis_func = traits.List(traits.Int(), minlen=2, maxlen=2, field='uweoptions.basfcn', desc='Number of basis functions to use for each dimension')
    est_reg_order = traits.Range(low=0, high=3, field='uweoptions.regorder', desc='This parameter determines how to balance the compromise between likelihood maximization and smoothness maximization of the estimated field.')
    est_reg_factor = traits.ListInt([100000], field='uweoptions.lambda', minlen=1, maxlen=1, usedefault=True, desc='Regularisation factor. Default: 100000 (medium).')
    est_jacobian_deformations = traits.Bool(field='uweoptions.jm', desc='Jacobian deformations. In theory a good idea to include them,  in practice a bad idea. Default: No.')
    est_first_order_effects = traits.List(traits.Int(), minlen=1, maxlen=6, field='uweoptions.fot', desc='First order effects should only depend on pitch and roll, i.e. [4 5]')
    est_second_order_effects = traits.List(traits.Int(), minlen=1, maxlen=6, field='uweoptions.sot', desc='List of second order terms to model second derivatives of.')
    est_unwarp_fwhm = traits.Range(low=0.0, field='uweoptions.uwfwhm', desc='gaussian smoothing kernel width for unwarp')
    est_re_est_mov_par = traits.Bool(field='uweoptions.rem', desc='Re-estimate movement parameters at each unwarping iteration.')
    est_num_of_iterations = traits.ListInt([5], field='uweoptions.noi', minlen=1, maxlen=1, usedefault=True, desc='Number of iterations.')
    est_taylor_expansion_point = traits.String('Average', field='uweoptions.expround', usedefault=True, desc='Point in position space to perform Taylor-expansion around.')
    reslice_which = traits.ListInt([2, 1], field='uwroptions.uwwhich', minlen=2, maxlen=2, usedefault=True, desc='determines which images to reslice')
    reslice_interp = traits.Range(low=0, high=7, field='uwroptions.rinterp', desc='degree of b-spline used for interpolation')
    reslice_wrap = traits.List(traits.Int(), minlen=3, maxlen=3, field='uwroptions.wrap', desc='Check if interpolation should wrap in [x,y,z]')
    reslice_mask = traits.Bool(field='uwroptions.mask', desc='True/False mask output image')
    out_prefix = traits.String('u', field='uwroptions.prefix', usedefault=True, desc='realigned and unwarped output prefix')