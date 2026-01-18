import logging
import numpy as np
from ..core import Format, image_as_uint
from ._freeimage import fi, IO_FLAGS
from .freeimage import FreeimageFormat
class GifFormat(FreeimageMulti):
    """A format for reading and writing static and animated GIF, based
    on the Freeimage library.

    Images read with this format are always RGBA. Currently,
    the alpha channel is ignored when saving RGB images with this
    format.

    The freeimage plugin requires a `freeimage` binary. If this binary
    is not available on the system, it can be downloaded by either

    - the command line script ``imageio_download_bin freeimage``
    - the Python method ``imageio.plugins.freeimage.download()``

    Parameters for reading
    ----------------------
    playback : bool
        'Play' the GIF to generate each frame (as 32bpp) instead of
        returning raw frame data when loading. Default True.

    Parameters for saving
    ---------------------
    loop : int
        The number of iterations. Default 0 (meaning loop indefinitely)
    duration : {float, list}
        The duration (in seconds) of each frame. Either specify one value
        that is used for all frames, or one value for each frame.
        Note that in the GIF format the duration/delay is expressed in
        hundredths of a second, which limits the precision of the duration.
    fps : float
        The number of frames per second. If duration is not given, the
        duration for each frame is set to 1/fps. Default 10.
    palettesize : int
        The number of colors to quantize the image to. Is rounded to
        the nearest power of two. Default 256.
    quantizer : {'wu', 'nq'}
        The quantization algorithm:
            * wu - Wu, Xiaolin, Efficient Statistical Computations for
              Optimal Color Quantization
            * nq (neuqant) - Dekker A. H., Kohonen neural networks for
              optimal color quantization
    subrectangles : bool
        If True, will try and optimize the GIF by storing only the
        rectangular parts of each frame that change with respect to the
        previous. Unfortunately, this option seems currently broken
        because FreeImage does not handle DisposalMethod correctly.
        Default False.
    """
    _fif = 25

    class Reader(FreeimageMulti.Reader):

        def _open(self, flags=0, playback=True):
            flags = int(flags)
            if playback:
                flags |= IO_FLAGS.GIF_PLAYBACK
            FreeimageMulti.Reader._open(self, flags)

        def _get_data(self, index):
            im, meta = FreeimageMulti.Reader._get_data(self, index)
            return (im, meta)

    class Writer(FreeimageMulti.Writer):

        def _open(self, flags=0, loop=0, duration=None, fps=10, palettesize=256, quantizer='Wu', subrectangles=False):
            if palettesize < 2 or palettesize > 256:
                raise ValueError('GIF quantize param must be 2..256')
            if palettesize not in [2, 4, 8, 16, 32, 64, 128, 256]:
                palettesize = 2 ** int(np.log2(128) + 0.999)
                logger.warning('Warning: palettesize (%r) modified to a factor of two between 2-256.' % palettesize)
            self._palettesize = palettesize
            self._quantizer = {'wu': 0, 'nq': 1}.get(quantizer.lower(), None)
            if self._quantizer is None:
                raise ValueError('Invalid quantizer, must be "wu" or "nq".')
            if duration is None:
                self._frametime = [int(1000 / float(fps) + 0.5)]
            elif isinstance(duration, list):
                self._frametime = [int(1000 * d) for d in duration]
            elif isinstance(duration, (float, int)):
                self._frametime = [int(1000 * duration)]
            else:
                raise ValueError('Invalid value for duration: %r' % duration)
            self._subrectangles = bool(subrectangles)
            self._prev_im = None
            FreeimageMulti.Writer._open(self, flags)
            self._meta = {}
            self._meta['ANIMATION'] = {'Loop': np.array([loop]).astype(np.uint32)}

        def _append_bitmap(self, im, meta, bitmap):
            meta = meta.copy()
            meta_a = meta['ANIMATION'] = {}
            if len(self._bm) == 0:
                meta.update(self._meta)
                meta_a = meta['ANIMATION']
            index = len(self._bm)
            if index < len(self._frametime):
                ft = self._frametime[index]
            else:
                ft = self._frametime[-1]
            meta_a['FrameTime'] = np.array([ft]).astype(np.uint32)
            if im.ndim == 3 and im.shape[-1] == 4:
                im = im[:, :, :3]
            im_uncropped = im
            if self._subrectangles and self._prev_im is not None:
                im, xy = self._get_sub_rectangles(self._prev_im, im)
                meta_a['DisposalMethod'] = np.array([1]).astype(np.uint8)
                meta_a['FrameLeft'] = np.array([xy[0]]).astype(np.uint16)
                meta_a['FrameTop'] = np.array([xy[1]]).astype(np.uint16)
            self._prev_im = im_uncropped
            sub2 = sub1 = bitmap
            sub1.allocate(im)
            sub1.set_image_data(im)
            if im.ndim == 3 and im.shape[-1] == 3:
                sub2 = sub1.quantize(self._quantizer, self._palettesize)
            sub2.set_meta_data(meta)
            return sub2

        def _get_sub_rectangles(self, prev, im):
            """
            Calculate the minimal rectangles that need updating each frame.
            Returns a two-element tuple containing the cropped images and a
            list of x-y positions.
            """
            diff = np.abs(im - prev)
            if diff.ndim == 3:
                diff = diff.sum(2)
            X = np.argwhere(diff.sum(0))
            Y = np.argwhere(diff.sum(1))
            if X.size and Y.size:
                x0, x1 = (int(X[0]), int(X[-1]) + 1)
                y0, y1 = (int(Y[0]), int(Y[-1]) + 1)
            else:
                x0, x1 = (0, 2)
                y0, y1 = (0, 2)
            return (im[y0:y1, x0:x1], (x0, y0))