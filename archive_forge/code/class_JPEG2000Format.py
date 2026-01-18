import logging
import threading
import numpy as np
from ..core import Format, image_as_uint
from ..core.request import URI_FILE, URI_BYTES
from .pillowmulti import GIFFormat, TIFFFormat  # noqa: E402, F401
class JPEG2000Format(PillowFormat):
    """See :mod:`imageio.plugins.pillow_legacy`"""

    class Reader(PillowFormat.Reader):

        def _open(self, pilmode=None, as_gray=False):
            return PillowFormat.Reader._open(self, pilmode=pilmode, as_gray=as_gray)

        def _get_file(self):
            if self.request.filename.startswith(('http://', 'https://')) or '.zip/' in self.request.filename.replace('\\', '/'):
                self._we_own_fp = True
                return open(self.request.get_local_filename(), 'rb')
            else:
                self._we_own_fp = False
                return self.request.get_file()

        def _get_data(self, index):
            im, info = PillowFormat.Reader._get_data(self, index)
            if 'exif' in info:
                from PIL.ExifTags import TAGS
                info['EXIF_MAIN'] = {}
                for tag, value in self._im._getexif().items():
                    decoded = TAGS.get(tag, tag)
                    info['EXIF_MAIN'][decoded] = value
            im = self._rotate(im, info)
            return (im, info)

        def _rotate(self, im, meta):
            """Use Orientation information from EXIF meta data to
            orient the image correctly. Similar code as in FreeImage plugin.
            """
            if self.request.kwargs.get('exifrotate', True):
                try:
                    ori = meta['EXIF_MAIN']['Orientation']
                except KeyError:
                    pass
                else:
                    if ori in [1, 2]:
                        pass
                    if ori in [3, 4]:
                        im = np.rot90(im, 2)
                    if ori in [5, 6]:
                        im = np.rot90(im, 3)
                    if ori in [7, 8]:
                        im = np.rot90(im)
                    if ori in [2, 4, 5, 7]:
                        im = np.fliplr(im)
            return im

    class Writer(PillowFormat.Writer):

        def _open(self, quality_mode='rates', quality=5, **kwargs):
            if quality_mode not in {'rates', 'dB'}:
                raise ValueError("Quality mode should be either 'rates' or 'dB'")
            quality = float(quality)
            if quality_mode == 'rates' and (quality < 1 or quality > 1000):
                raise ValueError('The quality value {} seems to be an invalid rate!'.format(quality))
            elif quality_mode == 'dB' and (quality < 15 or quality > 100):
                raise ValueError('The quality value {} seems to be an invalid PSNR!'.format(quality))
            kwargs['quality_mode'] = quality_mode
            kwargs['quality_layers'] = [quality]
            PillowFormat.Writer._open(self)
            self._meta.update(kwargs)

        def _append_data(self, im, meta):
            if im.ndim == 3 and im.shape[-1] == 4:
                raise IOError('The current implementation of JPEG 2000 does not support alpha channel.')
            im = image_as_uint(im, bitdepth=8)
            PillowFormat.Writer._append_data(self, im, meta)
            return