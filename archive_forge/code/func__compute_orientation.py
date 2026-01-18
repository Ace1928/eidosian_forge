import math
import numpy as np
import scipy.ndimage as ndi
from .._shared.utils import check_nD, _supported_float_type
from ..feature.util import DescriptorExtractor, FeatureDetector
from .._shared.filters import gaussian
from ..transform import rescale
from ..util import img_as_float
from ._sift import _local_max, _ori_distances, _update_histogram
def _compute_orientation(self, positions_oct, scales_oct, sigmas_oct, octaves, gaussian_scalespace):
    """Source: "Anatomy of the SIFT Method" Alg. 11
        Calculates the orientation of the gradient around every keypoint
        """
    gradient_space = []
    keypoint_indices = []
    keypoint_angles = []
    keypoint_octave = []
    orientations = np.zeros_like(sigmas_oct, dtype=self.float_dtype)
    key_count = 0
    for o, (octave, delta) in enumerate(zip(gaussian_scalespace, self.deltas)):
        gradient_space.append(np.gradient(octave))
        in_oct = octaves == o
        if not np.any(in_oct):
            continue
        positions = positions_oct[in_oct]
        scales = scales_oct[in_oct]
        sigmas = sigmas_oct[in_oct]
        oshape = octave.shape[:2]
        yx = positions / delta
        sigma = sigmas / delta
        radius = 3 * self.lambda_ori * sigma
        p_min = np.maximum(0, yx - radius[:, np.newaxis] + 0.5).astype(int)
        p_max = np.minimum(yx + radius[:, np.newaxis] + 0.5, (oshape[0] - 1, oshape[1] - 1)).astype(int)
        hist = np.empty(self.n_bins, dtype=self.float_dtype)
        avg_kernel = np.full((3,), 1 / 3, dtype=self.float_dtype)
        for k in range(len(yx)):
            hist[:] = 0
            r, c = np.meshgrid(np.arange(p_min[k, 0], p_max[k, 0] + 1), np.arange(p_min[k, 1], p_max[k, 1] + 1), indexing='ij', sparse=True)
            gradient_row = gradient_space[o][0][r, c, scales[k]]
            gradient_col = gradient_space[o][1][r, c, scales[k]]
            r = r.astype(self.float_dtype, copy=False)
            c = c.astype(self.float_dtype, copy=False)
            r -= yx[k, 0]
            c -= yx[k, 1]
            magnitude = np.sqrt(np.square(gradient_row) + np.square(gradient_col))
            theta = np.mod(np.arctan2(gradient_col, gradient_row), 2 * np.pi)
            kernel = np.exp(np.divide(r * r + c * c, -2 * (self.lambda_ori * sigma[k]) ** 2))
            bins = np.floor((theta / (2 * np.pi) * self.n_bins + 0.5) % self.n_bins).astype(int)
            np.add.at(hist, bins, kernel * magnitude)
            hist = np.concatenate((hist[-6:], hist, hist[:6]))
            for _ in range(6):
                hist = np.convolve(hist, avg_kernel, mode='same')
            hist = hist[6:-6]
            max_filter = ndi.maximum_filter(hist, [3], mode='wrap')
            maxima = np.nonzero(np.logical_and(hist >= self.c_max * np.max(hist), max_filter == hist))
            for c, m in enumerate(maxima[0]):
                neigh = np.arange(m - 1, m + 2) % len(hist)
                ori = (m + self._fit(hist[neigh]) + 0.5) * 2 * np.pi / self.n_bins
                if ori > np.pi:
                    ori -= 2 * np.pi
                if c == 0:
                    orientations[key_count] = ori
                else:
                    keypoint_indices.append(key_count)
                    keypoint_angles.append(ori)
                    keypoint_octave.append(o)
            key_count += 1
    self.positions = np.concatenate((positions_oct, positions_oct[keypoint_indices]))
    self.scales = np.concatenate((scales_oct, scales_oct[keypoint_indices]))
    self.sigmas = np.concatenate((sigmas_oct, sigmas_oct[keypoint_indices]))
    self.orientations = np.concatenate((orientations, keypoint_angles))
    self.octaves = np.concatenate((octaves, keypoint_octave))
    return gradient_space