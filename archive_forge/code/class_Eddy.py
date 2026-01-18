import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class Eddy(FSLCommand):
    """
    Interface for FSL eddy, a tool for estimating and correcting eddy
    currents induced distortions. `User guide
    <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide>`__ and
    `more info regarding acqp file
    <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/Faq#How_do_I_know_what_to_put_into_my_--acqp_file.3F>`_.

    Examples
    --------

    >>> from nipype.interfaces.fsl import Eddy

    Running eddy on a CPU using OpenMP:
    >>> eddy = Eddy()
    >>> eddy.inputs.in_file = 'epi.nii'
    >>> eddy.inputs.in_mask  = 'epi_mask.nii'
    >>> eddy.inputs.in_index = 'epi_index.txt'
    >>> eddy.inputs.in_acqp  = 'epi_acqp.txt'
    >>> eddy.inputs.in_bvec  = 'bvecs.scheme'
    >>> eddy.inputs.in_bval  = 'bvals.scheme'
    >>> eddy.cmdline          # doctest: +ELLIPSIS
    'eddy_openmp --flm=quadratic --ff=10.0 --acqp=epi_acqp.txt --bvals=bvals.scheme --bvecs=bvecs.scheme --imain=epi.nii --index=epi_index.txt --mask=epi_mask.nii --interp=spline --resamp=jac --niter=5 --nvoxhp=1000 --out=.../eddy_corrected --slm=none'

    Running eddy on an Nvidia GPU using cuda:
    >>> eddy.inputs.use_cuda = True
    >>> eddy.cmdline # doctest: +ELLIPSIS
    'eddy_cuda --flm=quadratic --ff=10.0 --acqp=epi_acqp.txt --bvals=bvals.scheme --bvecs=bvecs.scheme --imain=epi.nii --index=epi_index.txt --mask=epi_mask.nii --interp=spline --resamp=jac --niter=5 --nvoxhp=1000 --out=.../eddy_corrected --slm=none'

    Running eddy with slice-to-volume motion correction:
    >>> eddy.inputs.mporder = 6
    >>> eddy.inputs.slice2vol_niter = 5
    >>> eddy.inputs.slice2vol_lambda = 1
    >>> eddy.inputs.slice2vol_interp = 'trilinear'
    >>> eddy.inputs.slice_order = 'epi_slspec.txt'
    >>> eddy.cmdline          # doctest: +ELLIPSIS
    'eddy_cuda --flm=quadratic --ff=10.0 --acqp=epi_acqp.txt --bvals=bvals.scheme --bvecs=bvecs.scheme --imain=epi.nii --index=epi_index.txt --mask=epi_mask.nii --interp=spline --resamp=jac --mporder=6 --niter=5 --nvoxhp=1000 --out=.../eddy_corrected --s2v_interp=trilinear --s2v_lambda=1 --s2v_niter=5 --slspec=epi_slspec.txt --slm=none'
    >>> res = eddy.run()     # doctest: +SKIP

    """
    _cmd = 'eddy_openmp'
    input_spec = EddyInputSpec
    output_spec = EddyOutputSpec
    _num_threads = 1

    def __init__(self, **inputs):
        super(Eddy, self).__init__(**inputs)
        self.inputs.on_trait_change(self._num_threads_update, 'num_threads')
        if not isdefined(self.inputs.num_threads):
            self.inputs.num_threads = self._num_threads
        else:
            self._num_threads_update()
        self.inputs.on_trait_change(self._use_cuda, 'use_cuda')
        if isdefined(self.inputs.use_cuda):
            self._use_cuda()

    def _num_threads_update(self):
        self._num_threads = self.inputs.num_threads
        if not isdefined(self.inputs.num_threads):
            if 'OMP_NUM_THREADS' in self.inputs.environ:
                del self.inputs.environ['OMP_NUM_THREADS']
        else:
            self.inputs.environ['OMP_NUM_THREADS'] = str(self.inputs.num_threads)

    def _use_cuda(self):
        self._cmd = 'eddy_cuda' if self.inputs.use_cuda else 'eddy_openmp'

    def _run_interface(self, runtime):
        FSLDIR = os.getenv('FSLDIR', '')
        cmd = self._cmd
        if all((FSLDIR != '', cmd == 'eddy_openmp', not os.path.exists(os.path.join(FSLDIR, 'bin', cmd)))):
            self._cmd = 'eddy'
        runtime = super(Eddy, self)._run_interface(runtime)
        self._cmd = cmd
        return runtime

    def _format_arg(self, name, spec, value):
        if name == 'in_topup_fieldcoef':
            return spec.argstr % value.split('_fieldcoef')[0]
        if name == 'field':
            return spec.argstr % fname_presuffix(value, use_ext=False)
        if name == 'out_base':
            return spec.argstr % os.path.abspath(value)
        return super(Eddy, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_corrected'] = os.path.abspath('%s.nii.gz' % self.inputs.out_base)
        outputs['out_parameter'] = os.path.abspath('%s.eddy_parameters' % self.inputs.out_base)
        out_rotated_bvecs = os.path.abspath('%s.eddy_rotated_bvecs' % self.inputs.out_base)
        out_movement_rms = os.path.abspath('%s.eddy_movement_rms' % self.inputs.out_base)
        out_restricted_movement_rms = os.path.abspath('%s.eddy_restricted_movement_rms' % self.inputs.out_base)
        out_shell_alignment_parameters = os.path.abspath('%s.eddy_post_eddy_shell_alignment_parameters' % self.inputs.out_base)
        out_shell_pe_translation_parameters = os.path.abspath('%s.eddy_post_eddy_shell_PE_translation_parameters' % self.inputs.out_base)
        out_outlier_map = os.path.abspath('%s.eddy_outlier_map' % self.inputs.out_base)
        out_outlier_n_stdev_map = os.path.abspath('%s.eddy_outlier_n_stdev_map' % self.inputs.out_base)
        out_outlier_n_sqr_stdev_map = os.path.abspath('%s.eddy_outlier_n_sqr_stdev_map' % self.inputs.out_base)
        out_outlier_report = os.path.abspath('%s.eddy_outlier_report' % self.inputs.out_base)
        if isdefined(self.inputs.repol) and self.inputs.repol:
            out_outlier_free = os.path.abspath('%s.eddy_outlier_free_data' % self.inputs.out_base)
            if os.path.exists(out_outlier_free):
                outputs['out_outlier_free'] = out_outlier_free
        if isdefined(self.inputs.mporder) and self.inputs.mporder > 0:
            out_movement_over_time = os.path.abspath('%s.eddy_movement_over_time' % self.inputs.out_base)
            if os.path.exists(out_movement_over_time):
                outputs['out_movement_over_time'] = out_movement_over_time
        if isdefined(self.inputs.cnr_maps) and self.inputs.cnr_maps:
            out_cnr_maps = os.path.abspath('%s.eddy_cnr_maps.nii.gz' % self.inputs.out_base)
            if os.path.exists(out_cnr_maps):
                outputs['out_cnr_maps'] = out_cnr_maps
        if isdefined(self.inputs.residuals) and self.inputs.residuals:
            out_residuals = os.path.abspath('%s.eddy_residuals.nii.gz' % self.inputs.out_base)
            if os.path.exists(out_residuals):
                outputs['out_residuals'] = out_residuals
        if os.path.exists(out_rotated_bvecs):
            outputs['out_rotated_bvecs'] = out_rotated_bvecs
        if os.path.exists(out_movement_rms):
            outputs['out_movement_rms'] = out_movement_rms
        if os.path.exists(out_restricted_movement_rms):
            outputs['out_restricted_movement_rms'] = out_restricted_movement_rms
        if os.path.exists(out_shell_alignment_parameters):
            outputs['out_shell_alignment_parameters'] = out_shell_alignment_parameters
        if os.path.exists(out_shell_pe_translation_parameters):
            outputs['out_shell_pe_translation_parameters'] = out_shell_pe_translation_parameters
        if os.path.exists(out_outlier_map):
            outputs['out_outlier_map'] = out_outlier_map
        if os.path.exists(out_outlier_n_stdev_map):
            outputs['out_outlier_n_stdev_map'] = out_outlier_n_stdev_map
        if os.path.exists(out_outlier_n_sqr_stdev_map):
            outputs['out_outlier_n_sqr_stdev_map'] = out_outlier_n_sqr_stdev_map
        if os.path.exists(out_outlier_report):
            outputs['out_outlier_report'] = out_outlier_report
        return outputs