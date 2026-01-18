import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class GLMFitOutputSpec(TraitedSpec):
    glm_dir = Directory(exists=True, desc='output directory')
    beta_file = File(exists=True, desc='map of regression coefficients')
    error_file = File(desc='map of residual error')
    error_var_file = File(desc='map of residual error variance')
    error_stddev_file = File(desc='map of residual error standard deviation')
    estimate_file = File(desc='map of the estimated Y values')
    mask_file = File(desc='map of the mask used in the analysis')
    fwhm_file = File(desc='text file with estimated smoothness')
    dof_file = File(desc='text file with effective degrees-of-freedom for the analysis')
    gamma_file = OutputMultiPath(desc='map of contrast of regression coefficients')
    gamma_var_file = OutputMultiPath(desc='map of regression contrast variance')
    sig_file = OutputMultiPath(desc='map of F-test significance (in -log10p)')
    ftest_file = OutputMultiPath(desc='map of test statistic values')
    spatial_eigenvectors = File(desc='map of spatial eigenvectors from residual PCA')
    frame_eigenvectors = File(desc='matrix of frame eigenvectors from residual PCA')
    singular_values = File(desc='matrix singular values from residual PCA')
    svd_stats_file = File(desc='text file summarizing the residual PCA')
    k2p_file = File(desc='estimate of k2p parameter')
    bp_file = File(desc='Binding potential estimates')