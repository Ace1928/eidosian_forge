import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class GLMFit(FSCommand):
    """Use FreeSurfer's mri_glmfit to specify and estimate a general linear model.

    Examples
    --------
    >>> glmfit = GLMFit()
    >>> glmfit.inputs.in_file = 'functional.nii'
    >>> glmfit.inputs.one_sample = True
    >>> glmfit.cmdline == 'mri_glmfit --glmdir %s --y functional.nii --osgm'%os.getcwd()
    True

    """
    _cmd = 'mri_glmfit'
    input_spec = GLMFitInputSpec
    output_spec = GLMFitOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'surf':
            _si = self.inputs
            return spec.argstr % (_si.subject_id, _si.hemi, _si.surf_geo)
        return super(GLMFit, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.glm_dir):
            glmdir = os.getcwd()
        else:
            glmdir = os.path.abspath(self.inputs.glm_dir)
        outputs['glm_dir'] = glmdir
        if isdefined(self.inputs.nii_gz):
            ext = 'nii.gz'
        elif isdefined(self.inputs.nii):
            ext = 'nii'
        else:
            ext = 'mgh'
        outputs['beta_file'] = os.path.join(glmdir, f'beta.{ext}')
        outputs['error_var_file'] = os.path.join(glmdir, f'rvar.{ext}')
        outputs['error_stddev_file'] = os.path.join(glmdir, f'rstd.{ext}')
        outputs['mask_file'] = os.path.join(glmdir, f'mask.{ext}')
        outputs['fwhm_file'] = os.path.join(glmdir, 'fwhm.dat')
        outputs['dof_file'] = os.path.join(glmdir, 'dof.dat')
        if self.inputs.save_residual:
            outputs['error_file'] = os.path.join(glmdir, f'eres.{ext}')
        if self.inputs.save_estimate:
            outputs['estimate_file'] = os.path.join(glmdir, f'yhat.{ext}')
        if any((self.inputs.mrtm1, self.inputs.mrtm2, self.inputs.logan)):
            outputs['bp_file'] = os.path.join(glmdir, f'bp.{ext}')
        if self.inputs.mrtm1:
            outputs['k2p_file'] = os.path.join(glmdir, 'k2prime.dat')
        contrasts = []
        if isdefined(self.inputs.contrast):
            for c in self.inputs.contrast:
                if split_filename(c)[2] in ['.mat', '.dat', '.mtx', '.con']:
                    contrasts.append(split_filename(c)[1])
                else:
                    contrasts.append(os.path.split(c)[1])
        elif isdefined(self.inputs.one_sample) and self.inputs.one_sample:
            contrasts = ['osgm']
        outputs['sig_file'] = [os.path.join(glmdir, c, f'sig.{ext}') for c in contrasts]
        outputs['ftest_file'] = [os.path.join(glmdir, c, f'F.{ext}') for c in contrasts]
        outputs['gamma_file'] = [os.path.join(glmdir, c, f'gamma.{ext}') for c in contrasts]
        outputs['gamma_var_file'] = [os.path.join(glmdir, c, f'gammavar.{ext}') for c in contrasts]
        if isdefined(self.inputs.pca) and self.inputs.pca:
            pcadir = os.path.join(glmdir, 'pca-eres')
            outputs['spatial_eigenvectors'] = os.path.join(pcadir, f'v.{ext}')
            outputs['frame_eigenvectors'] = os.path.join(pcadir, 'u.mtx')
            outputs['singluar_values'] = os.path.join(pcadir, 'sdiag.mat')
            outputs['svd_stats_file'] = os.path.join(pcadir, 'stats.dat')
        return outputs

    def _gen_filename(self, name):
        if name == 'glm_dir':
            return os.getcwd()
        return None