import logging
import numpy as np
from ..core import Format, image_as_uint
from ._freeimage import fi, IO_FLAGS
from .freeimage import FreeimageFormat
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