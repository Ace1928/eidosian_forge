import os
from copy import deepcopy
from nibabel import load, funcs, Nifti1Image
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list, save_json, split_filename
from ..utils.misc import find_indices, normalize_mc_params
from .. import logging, config
def _detect_outliers_core(self, imgfile, motionfile, runidx, cwd=None):
    """
        Core routine for detecting outliers
        """
    from scipy import signal
    if not cwd:
        cwd = os.getcwd()
    if isinstance(imgfile, (str, bytes)):
        nim = load(imgfile)
    elif isinstance(imgfile, list):
        if len(imgfile) == 1:
            nim = load(imgfile[0])
        else:
            images = [load(f) for f in imgfile]
            nim = funcs.concat_images(images)
    x, y, z, timepoints = nim.shape
    data = nim.get_fdata(dtype=np.float32)
    affine = nim.affine
    g = np.zeros((timepoints, 1))
    masktype = self.inputs.mask_type
    if masktype == 'spm_global':
        iflogger.debug('art: using spm global')
        intersect_mask = self.inputs.intersect_mask
        if intersect_mask:
            mask = np.ones((x, y, z), dtype=bool)
            for t0 in range(timepoints):
                vol = data[:, :, :, t0]
                mask_tmp = vol > np.nanmean(vol) / self.inputs.global_threshold
                mask = mask * mask_tmp
            for t0 in range(timepoints):
                vol = data[:, :, :, t0]
                g[t0] = np.nanmean(vol[mask])
            if len(find_indices(mask)) < np.prod((x, y, z)) / 10:
                intersect_mask = False
                g = np.zeros((timepoints, 1))
        if not intersect_mask:
            iflogger.info('not intersect_mask is True')
            mask = np.zeros((x, y, z, timepoints))
            for t0 in range(timepoints):
                vol = data[:, :, :, t0]
                mask_tmp = vol > np.nanmean(vol) / self.inputs.global_threshold
                mask[:, :, :, t0] = mask_tmp
                g[t0] = np.nansum(vol * mask_tmp) / np.nansum(mask_tmp)
    elif masktype == 'file':
        maskimg = load(self.inputs.mask_file)
        mask = maskimg.get_fdata(dtype=np.float32)
        affine = maskimg.affine
        mask = mask > 0.5
        for t0 in range(timepoints):
            vol = data[:, :, :, t0]
            g[t0] = np.nanmean(vol[mask])
    elif masktype == 'thresh':
        for t0 in range(timepoints):
            vol = data[:, :, :, t0]
            mask = vol > self.inputs.mask_threshold
            g[t0] = np.nanmean(vol[mask])
    else:
        mask = np.ones((x, y, z))
        g = np.nanmean(data[mask > 0, :], 1)
    gz = signal.detrend(g, axis=0)
    if self.inputs.use_differences[1]:
        gz = np.concatenate((np.zeros((1, 1)), np.diff(gz, n=1, axis=0)), axis=0)
    gz = (gz - np.mean(gz)) / np.std(gz)
    iidx = find_indices(abs(gz) > self.inputs.zintensity_threshold)
    mc_in = np.loadtxt(motionfile)
    mc = deepcopy(mc_in)
    artifactfile, intensityfile, statsfile, normfile, plotfile, displacementfile, maskfile = self._get_output_filenames(imgfile, cwd)
    mask_img = Nifti1Image(mask.astype(np.uint8), affine)
    mask_img.to_filename(maskfile)
    if self.inputs.use_norm:
        brain_pts = None
        if self.inputs.bound_by_brainmask:
            voxel_coords = np.nonzero(mask)
            coords = np.vstack((voxel_coords[0], np.vstack((voxel_coords[1], voxel_coords[2])))).T
            brain_pts = np.dot(affine, np.hstack((coords, np.ones((coords.shape[0], 1)))).T)
        normval, displacement = _calc_norm(mc, self.inputs.use_differences[0], self.inputs.parameter_source, brain_pts=brain_pts)
        tidx = find_indices(normval > self.inputs.norm_threshold)
        ridx = find_indices(normval < 0)
        if displacement is not None:
            dmap = np.zeros((x, y, z, timepoints), dtype=np.float64)
            for i in range(timepoints):
                dmap[voxel_coords[0], voxel_coords[1], voxel_coords[2], i] = displacement[i, :]
            dimg = Nifti1Image(dmap, affine)
            dimg.to_filename(displacementfile)
    else:
        if self.inputs.use_differences[0]:
            mc = np.concatenate((np.zeros((1, 6)), np.diff(mc_in, n=1, axis=0)), axis=0)
        traval = mc[:, 0:3]
        rotval = mc[:, 3:6]
        tidx = find_indices(np.sum(abs(traval) > self.inputs.translation_threshold, 1) > 0)
        ridx = find_indices(np.sum(abs(rotval) > self.inputs.rotation_threshold, 1) > 0)
    outliers = np.unique(np.union1d(iidx, np.union1d(tidx, ridx)))
    np.savetxt(artifactfile, outliers, fmt=b'%d', delimiter=' ')
    np.savetxt(intensityfile, g, fmt=b'%.2f', delimiter=' ')
    if self.inputs.use_norm:
        np.savetxt(normfile, normval, fmt=b'%.4f', delimiter=' ')
    if isdefined(self.inputs.save_plot) and self.inputs.save_plot:
        import matplotlib
        matplotlib.use(config.get('execution', 'matplotlib_backend'))
        import matplotlib.pyplot as plt
        fig = plt.figure()
        if isdefined(self.inputs.use_norm) and self.inputs.use_norm:
            plt.subplot(211)
        else:
            plt.subplot(311)
        self._plot_outliers_with_wave(gz, iidx, 'Intensity')
        if isdefined(self.inputs.use_norm) and self.inputs.use_norm:
            plt.subplot(212)
            self._plot_outliers_with_wave(normval, np.union1d(tidx, ridx), 'Norm (mm)')
        else:
            diff = ''
            if self.inputs.use_differences[0]:
                diff = 'diff'
            plt.subplot(312)
            self._plot_outliers_with_wave(traval, tidx, 'Translation (mm)' + diff)
            plt.subplot(313)
            self._plot_outliers_with_wave(rotval, ridx, 'Rotation (rad)' + diff)
        plt.savefig(plotfile)
        plt.close(fig)
    motion_outliers = np.union1d(tidx, ridx)
    stats = [{'motion_file': motionfile, 'functional_file': imgfile}, {'common_outliers': len(np.intersect1d(iidx, motion_outliers)), 'intensity_outliers': len(np.setdiff1d(iidx, motion_outliers)), 'motion_outliers': len(np.setdiff1d(motion_outliers, iidx))}, {'motion': [{'using differences': self.inputs.use_differences[0]}, {'mean': np.mean(mc_in, axis=0).tolist(), 'min': np.min(mc_in, axis=0).tolist(), 'max': np.max(mc_in, axis=0).tolist(), 'std': np.std(mc_in, axis=0).tolist()}]}, {'intensity': [{'using differences': self.inputs.use_differences[1]}, {'mean': np.mean(gz, axis=0).tolist(), 'min': np.min(gz, axis=0).tolist(), 'max': np.max(gz, axis=0).tolist(), 'std': np.std(gz, axis=0).tolist()}]}]
    if self.inputs.use_norm:
        stats.insert(3, {'motion_norm': {'mean': np.mean(normval, axis=0).tolist(), 'min': np.min(normval, axis=0).tolist(), 'max': np.max(normval, axis=0).tolist(), 'std': np.std(normval, axis=0).tolist()}})
    save_json(statsfile, stats)