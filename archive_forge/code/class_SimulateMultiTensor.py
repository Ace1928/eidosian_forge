from multiprocessing import Pool, cpu_count
import os.path as op
import numpy as np
import nibabel as nb
from ... import logging
from ..base import (
from .base import DipyBaseInterface
class SimulateMultiTensor(DipyBaseInterface):
    """
    Interface to MultiTensor model simulator in dipy
    http://nipy.org/dipy/examples_built/simulate_multi_tensor.html

    Example
    -------

    >>> import nipype.interfaces.dipy as dipy
    >>> sim = dipy.SimulateMultiTensor()
    >>> sim.inputs.in_dirs = ['fdir00.nii', 'fdir01.nii']
    >>> sim.inputs.in_frac = ['ffra00.nii', 'ffra01.nii']
    >>> sim.inputs.in_vfms = ['tpm_00.nii.gz', 'tpm_01.nii.gz',
    ...                       'tpm_02.nii.gz']
    >>> sim.inputs.baseline = 'b0.nii'
    >>> sim.inputs.in_bvec = 'bvecs'
    >>> sim.inputs.in_bval = 'bvals'
    >>> sim.run()                                   # doctest: +SKIP
    """
    input_spec = SimulateMultiTensorInputSpec
    output_spec = SimulateMultiTensorOutputSpec

    def _run_interface(self, runtime):
        from dipy.core.gradients import gradient_table
        if isdefined(self.inputs.in_bval) and isdefined(self.inputs.in_bvec):
            bvals = np.loadtxt(self.inputs.in_bval)
            bvecs = np.loadtxt(self.inputs.in_bvec).T
            gtab = gradient_table(bvals, bvecs)
        else:
            gtab = _generate_gradients(self.inputs.num_dirs, self.inputs.bvalues)
        ndirs = len(gtab.bvals)
        np.savetxt(op.abspath(self.inputs.out_bvec), gtab.bvecs.T)
        np.savetxt(op.abspath(self.inputs.out_bval), gtab.bvals)
        b0_im = nb.load(self.inputs.baseline)
        hdr = b0_im.header
        shape = b0_im.shape
        aff = b0_im.affine
        nsticks = len(self.inputs.in_dirs)
        if len(self.inputs.in_frac) != nsticks:
            raise RuntimeError('Number of sticks and their volume fractions must match.')
        nballs = len(self.inputs.in_vfms)
        vfs = np.squeeze(nb.concat_images(self.inputs.in_vfms).dataobj)
        if nballs == 1:
            vfs = vfs[..., np.newaxis]
        total_vf = np.sum(vfs, axis=3)
        if isdefined(self.inputs.in_mask):
            msk = np.asanyarray(nb.load(self.inputs.in_mask).dataobj)
            msk[msk > 0.0] = 1.0
            msk[msk < 1.0] = 0.0
        else:
            msk = np.zeros(shape)
            msk[total_vf > 0.0] = 1.0
        msk = np.clip(msk, 0.0, 1.0)
        nvox = len(msk[msk > 0])
        ffsim = nb.concat_images(self.inputs.in_frac)
        ffs = np.nan_to_num(np.squeeze(ffsim.dataobj))
        ffs = np.clip(ffs, 0.0, 1.0)
        if nsticks == 1:
            ffs = ffs[..., np.newaxis]
        for i in range(nsticks):
            ffs[..., i] *= msk
        total_ff = np.sum(ffs, axis=3)
        for i in range(1, nsticks):
            if np.any(total_ff > 1.0):
                errors = np.zeros_like(total_ff)
                errors[total_ff > 1.0] = total_ff[total_ff > 1.0] - 1.0
                ffs[..., i] -= errors
                ffs[ffs < 0.0] = 0.0
            total_ff = np.sum(ffs, axis=3)
        for i in range(vfs.shape[-1]):
            vfs[..., i] -= total_ff
        vfs = np.clip(vfs, 0.0, 1.0)
        fractions = np.concatenate((ffs, vfs), axis=3)
        nb.Nifti1Image(fractions, aff, None).to_filename('fractions.nii.gz')
        nb.Nifti1Image(np.sum(fractions, axis=3), aff, None).to_filename('total_vf.nii.gz')
        mhdr = hdr.copy()
        mhdr.set_data_dtype(np.uint8)
        mhdr.set_xyzt_units('mm', 'sec')
        nb.Nifti1Image(msk, aff, mhdr).to_filename(op.abspath(self.inputs.out_mask))
        fracs = fractions[msk > 0]
        dirs = None
        for i in range(nsticks):
            f = self.inputs.in_dirs[i]
            fd = np.nan_to_num(nb.load(f).dataobj)
            w = np.linalg.norm(fd, axis=3)[..., np.newaxis]
            w[w < np.finfo(float).eps] = 1.0
            fd /= w
            if dirs is None:
                dirs = fd[msk > 0].copy()
            else:
                dirs = np.hstack((dirs, fd[msk > 0]))
        for d in range(nballs):
            fd = np.random.randn(nvox, 3)
            w = np.linalg.norm(fd, axis=1)
            fd[w < np.finfo(float).eps, ...] = np.array([1.0, 0.0, 0.0])
            w[w < np.finfo(float).eps] = 1.0
            fd /= w[..., np.newaxis]
            dirs = np.hstack((dirs, fd))
        sf_evals = list(self.inputs.diff_sf)
        ba_evals = list(self.inputs.diff_iso)
        mevals = [sf_evals] * nsticks + [[ba_evals[d]] * 3 for d in range(nballs)]
        b0 = b0_im.get_fdata()[msk > 0]
        args = []
        for i in range(nvox):
            args.append({'fractions': fracs[i, ...].tolist(), 'sticks': [tuple(dirs[i, j:j + 3]) for j in range(nsticks + nballs)], 'gradients': gtab, 'mevals': mevals, 'S0': b0[i], 'snr': self.inputs.snr})
        n_proc = self.inputs.n_proc
        if n_proc == 0:
            n_proc = cpu_count()
        try:
            pool = Pool(processes=n_proc, maxtasksperchild=50)
        except TypeError:
            pool = Pool(processes=n_proc)
        IFLOGGER.info('Starting simulation of %d voxels, %d diffusion directions.', len(args), ndirs)
        result = np.array(pool.map(_compute_voxel, args))
        if np.shape(result)[1] != ndirs:
            raise RuntimeError('Computed directions do not match numberof b-values.')
        signal = np.zeros((shape[0], shape[1], shape[2], ndirs))
        signal[msk > 0] = result
        simhdr = hdr.copy()
        simhdr.set_data_dtype(np.float32)
        simhdr.set_xyzt_units('mm', 'sec')
        nb.Nifti1Image(signal.astype(np.float32), aff, simhdr).to_filename(op.abspath(self.inputs.out_file))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        outputs['out_mask'] = op.abspath(self.inputs.out_mask)
        outputs['out_bvec'] = op.abspath(self.inputs.out_bvec)
        outputs['out_bval'] = op.abspath(self.inputs.out_bval)
        return outputs