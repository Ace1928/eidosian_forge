import math
import os
import logging
from pathlib import Path
import warnings
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import matplotlib as mpl
from matplotlib import _api, cbook, cm
from matplotlib import _image
from matplotlib._image import *
import matplotlib.artist as martist
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.colors as mcolors
from matplotlib.transforms import (
def _make_image(self, A, in_bbox, out_bbox, clip_bbox, magnification=1.0, unsampled=False, round_to_pixel_border=True):
    """
        Normalize, rescale, and colormap the image *A* from the given *in_bbox*
        (in data space), to the given *out_bbox* (in pixel space) clipped to
        the given *clip_bbox* (also in pixel space), and magnified by the
        *magnification* factor.

        *A* may be a greyscale image (M, N) with a dtype of `~numpy.float32`,
        `~numpy.float64`, `~numpy.float128`, `~numpy.uint16` or `~numpy.uint8`,
        or an (M, N, 4) RGBA image with a dtype of `~numpy.float32`,
        `~numpy.float64`, `~numpy.float128`, or `~numpy.uint8`.

        If *unsampled* is True, the image will not be scaled, but an
        appropriate affine transformation will be returned instead.

        If *round_to_pixel_border* is True, the output image size will be
        rounded to the nearest pixel boundary.  This makes the images align
        correctly with the axes.  It should not be used if exact scaling is
        needed, such as for `FigureImage`.

        Returns
        -------
        image : (M, N, 4) `numpy.uint8` array
            The RGBA image, resampled unless *unsampled* is True.
        x, y : float
            The upper left corner where the image should be drawn, in pixel
            space.
        trans : `~matplotlib.transforms.Affine2D`
            The affine transformation from image to pixel space.
        """
    if A is None:
        raise RuntimeError('You must first set the image array or the image attribute')
    if A.size == 0:
        raise RuntimeError("_make_image must get a non-empty image. Your Artist's draw method must filter before this method is called.")
    clipped_bbox = Bbox.intersection(out_bbox, clip_bbox)
    if clipped_bbox is None:
        return (None, 0, 0, None)
    out_width_base = clipped_bbox.width * magnification
    out_height_base = clipped_bbox.height * magnification
    if out_width_base == 0 or out_height_base == 0:
        return (None, 0, 0, None)
    if self.origin == 'upper':
        t0 = Affine2D().translate(0, -A.shape[0]).scale(1, -1)
    else:
        t0 = IdentityTransform()
    t0 += Affine2D().scale(in_bbox.width / A.shape[1], in_bbox.height / A.shape[0]).translate(in_bbox.x0, in_bbox.y0) + self.get_transform()
    t = t0 + Affine2D().translate(-clipped_bbox.x0, -clipped_bbox.y0).scale(magnification)
    if not unsampled and t.is_affine and round_to_pixel_border and (out_width_base % 1.0 != 0.0 or out_height_base % 1.0 != 0.0):
        out_width = math.ceil(out_width_base)
        out_height = math.ceil(out_height_base)
        extra_width = (out_width - out_width_base) / out_width_base
        extra_height = (out_height - out_height_base) / out_height_base
        t += Affine2D().scale(1.0 + extra_width, 1.0 + extra_height)
    else:
        out_width = int(out_width_base)
        out_height = int(out_height_base)
    out_shape = (out_height, out_width)
    if not unsampled:
        if not (A.ndim == 2 or (A.ndim == 3 and A.shape[-1] in (3, 4))):
            raise ValueError(f'Invalid shape {A.shape} for image data')
        if A.ndim == 2 and self._interpolation_stage != 'rgba':
            a_min = A.min()
            a_max = A.max()
            if a_min is np.ma.masked:
                a_min, a_max = (np.int32(0), np.int32(1))
            if A.dtype.kind == 'f':
                scaled_dtype = np.dtype(np.float64 if A.dtype.itemsize > 4 else np.float32)
                if scaled_dtype.itemsize < A.dtype.itemsize:
                    _api.warn_external(f'Casting input data from {A.dtype} to {scaled_dtype} for imshow.')
            else:
                da = a_max.astype(np.float64) - a_min.astype(np.float64)
                scaled_dtype = np.float64 if da > 100000000.0 else np.float32
            A_scaled = np.array(A, dtype=scaled_dtype)
            self.norm.autoscale_None(A)
            dv = np.float64(self.norm.vmax) - np.float64(self.norm.vmin)
            vmid = np.float64(self.norm.vmin) + dv / 2
            fact = 10000000.0 if scaled_dtype == np.float64 else 10000.0
            newmin = vmid - dv * fact
            if newmin < a_min:
                newmin = None
            else:
                a_min = np.float64(newmin)
            newmax = vmid + dv * fact
            if newmax > a_max:
                newmax = None
            else:
                a_max = np.float64(newmax)
            if newmax is not None or newmin is not None:
                np.clip(A_scaled, newmin, newmax, out=A_scaled)
            offset = 0.1
            frac = 0.8
            vmin, vmax = (self.norm.vmin, self.norm.vmax)
            if vmin is np.ma.masked:
                vmin, vmax = (a_min, a_max)
            vrange = np.array([vmin, vmax], dtype=scaled_dtype)
            A_scaled -= a_min
            vrange -= a_min
            a_min = a_min.astype(scaled_dtype).item()
            a_max = a_max.astype(scaled_dtype).item()
            if a_min != a_max:
                A_scaled /= (a_max - a_min) / frac
                vrange /= (a_max - a_min) / frac
            A_scaled += offset
            vrange += offset
            A_resampled = _resample(self, A_scaled, out_shape, t)
            del A_scaled
            A_resampled -= offset
            vrange -= offset
            if a_min != a_max:
                A_resampled *= (a_max - a_min) / frac
                vrange *= (a_max - a_min) / frac
            A_resampled += a_min
            vrange += a_min
            if isinstance(self.norm, mcolors.NoNorm):
                A_resampled = A_resampled.astype(A.dtype)
            mask = np.where(A.mask, np.float32(np.nan), np.float32(1)) if A.mask.shape == A.shape else np.ones_like(A, np.float32)
            out_alpha = _resample(self, mask, out_shape, t, resample=True)
            del mask
            out_mask = np.isnan(out_alpha)
            out_alpha[out_mask] = 1
            alpha = self.get_alpha()
            if alpha is not None and np.ndim(alpha) > 0:
                out_alpha *= _resample(self, alpha, out_shape, t, resample=True)
            resampled_masked = np.ma.masked_array(A_resampled, out_mask)
            s_vmin, s_vmax = vrange
            if isinstance(self.norm, mcolors.LogNorm) and s_vmin <= 0:
                s_vmin = np.finfo(scaled_dtype).eps
            with self.norm.callbacks.blocked(), cbook._setattr_cm(self.norm, vmin=s_vmin, vmax=s_vmax):
                output = self.norm(resampled_masked)
        else:
            if A.ndim == 2:
                self.norm.autoscale_None(A)
                A = self.to_rgba(A)
            if A.shape[2] == 3:
                A = _rgb_to_rgba(A)
            alpha = self._get_scalar_alpha()
            output_alpha = _resample(self, A[..., 3], out_shape, t, alpha=alpha)
            output = _resample(self, _rgb_to_rgba(A[..., :3]), out_shape, t, alpha=alpha)
            output[..., 3] = output_alpha
        output = self.to_rgba(output, bytes=True, norm=False)
        if A.ndim == 2:
            alpha = self._get_scalar_alpha()
            alpha_channel = output[:, :, 3]
            alpha_channel[:] = alpha_channel.astype(np.float32) * out_alpha * alpha
    else:
        if self._imcache is None:
            self._imcache = self.to_rgba(A, bytes=True, norm=A.ndim == 2)
        output = self._imcache
        subset = TransformedBbox(clip_bbox, t0.inverted()).frozen()
        output = output[int(max(subset.ymin, 0)):int(min(subset.ymax + 1, output.shape[0])), int(max(subset.xmin, 0)):int(min(subset.xmax + 1, output.shape[1]))]
        t = Affine2D().translate(int(max(subset.xmin, 0)), int(max(subset.ymin, 0))) + t
    return (output, clipped_bbox.x0, clipped_bbox.y0, t)