import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class TOPUPInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, desc='name of 4D file with images', argstr='--imain=%s')
    encoding_file = File(exists=True, mandatory=True, xor=['encoding_direction'], desc='name of text file with PE directions/times', argstr='--datain=%s')
    encoding_direction = traits.List(traits.Enum('y', 'x', 'z', 'x-', 'y-', 'z-'), mandatory=True, xor=['encoding_file'], requires=['readout_times'], argstr='--datain=%s', desc='encoding direction for automatic generation of encoding_file')
    readout_times = InputMultiPath(traits.Float, requires=['encoding_direction'], xor=['encoding_file'], mandatory=True, desc='readout times (dwell times by # phase-encode steps minus 1)')
    out_base = File(desc='base-name of output files (spline coefficients (Hz) and movement parameters)', name_source=['in_file'], name_template='%s_base', argstr='--out=%s', hash_files=False)
    out_field = File(argstr='--fout=%s', hash_files=False, name_source=['in_file'], name_template='%s_field', desc='name of image file with field (Hz)')
    out_warp_prefix = traits.Str('warpfield', argstr='--dfout=%s', hash_files=False, desc='prefix for the warpfield images (in mm)', usedefault=True)
    out_mat_prefix = traits.Str('xfm', argstr='--rbmout=%s', hash_files=False, desc='prefix for the realignment matrices', usedefault=True)
    out_jac_prefix = traits.Str('jac', argstr='--jacout=%s', hash_files=False, desc='prefix for the warpfield images', usedefault=True)
    out_corrected = File(argstr='--iout=%s', hash_files=False, name_source=['in_file'], name_template='%s_corrected', desc='name of 4D image file with unwarped images')
    out_logfile = File(argstr='--logout=%s', desc='name of log-file', name_source=['in_file'], name_template='%s_topup.log', keep_extension=True, hash_files=False)
    warp_res = traits.Float(argstr='--warpres=%f', desc='(approximate) resolution (in mm) of warp basis for the different sub-sampling levels')
    subsamp = traits.Int(argstr='--subsamp=%d', desc='sub-sampling scheme')
    fwhm = traits.Float(argstr='--fwhm=%f', desc='FWHM (in mm) of gaussian smoothing kernel')
    config = traits.String('b02b0.cnf', argstr='--config=%s', usedefault=True, desc='Name of config file specifying command line arguments')
    max_iter = traits.Int(argstr='--miter=%d', desc='max # of non-linear iterations')
    reg_lambda = traits.Float(argstr='--lambda=%0.f', desc='Weight of regularisation, default depending on --ssqlambda and --regmod switches.')
    ssqlambda = traits.Enum(1, 0, argstr='--ssqlambda=%d', desc='Weight lambda by the current value of the ssd. If used (=1), the effective weight of regularisation term becomes higher for the initial iterations, therefore initial steps are a little smoother than they would without weighting. This reduces the risk of finding a local minimum.')
    regmod = traits.Enum('bending_energy', 'membrane_energy', argstr='--regmod=%s', desc='Regularisation term implementation. Defaults to bending_energy. Note that the two functions have vastly different scales. The membrane energy is based on the first derivatives and the bending energy on the second derivatives. The second derivatives will typically be much smaller than the first derivatives, so input lambda will have to be larger for bending_energy to yield approximately the same level of regularisation.')
    estmov = traits.Enum(1, 0, argstr='--estmov=%d', desc='estimate movements if set')
    minmet = traits.Enum(0, 1, argstr='--minmet=%d', desc='Minimisation method 0=Levenberg-Marquardt, 1=Scaled Conjugate Gradient')
    splineorder = traits.Int(argstr='--splineorder=%d', desc='order of spline, 2->Qadratic spline, 3->Cubic spline')
    numprec = traits.Enum('double', 'float', argstr='--numprec=%s', desc='Precision for representing Hessian, double or float.')
    interp = traits.Enum('spline', 'linear', argstr='--interp=%s', desc='Image interpolation model, linear or spline.')
    scale = traits.Enum(0, 1, argstr='--scale=%d', desc='If set (=1), the images are individually scaled to a common mean')
    regrid = traits.Enum(1, 0, argstr='--regrid=%d', desc='If set (=1), the calculations are done in a different grid')