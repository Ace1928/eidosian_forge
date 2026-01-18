from __future__ import annotations
import itertools
import os
import struct
from . import (
from ._binary import i16be as i16
from ._binary import o32le
class MpoImageFile(JpegImagePlugin.JpegImageFile):
    format = 'MPO'
    format_description = 'MPO (CIPA DC-007)'
    _close_exclusive_fp_after_loading = False

    def _open(self):
        self.fp.seek(0)
        JpegImagePlugin.JpegImageFile._open(self)
        self._after_jpeg_open()

    def _after_jpeg_open(self, mpheader=None):
        self._initial_size = self.size
        self.mpinfo = mpheader if mpheader is not None else self._getmp()
        self.n_frames = self.mpinfo[45057]
        self.__mpoffsets = [mpent['DataOffset'] + self.info['mpoffset'] for mpent in self.mpinfo[45058]]
        self.__mpoffsets[0] = 0
        assert self.n_frames == len(self.__mpoffsets)
        del self.info['mpoffset']
        self.is_animated = self.n_frames > 1
        self._fp = self.fp
        self._fp.seek(self.__mpoffsets[0])
        self.__frame = 0
        self.offset = 0
        self.readonly = 1

    def load_seek(self, pos):
        self._fp.seek(pos)

    def seek(self, frame):
        if not self._seek_check(frame):
            return
        self.fp = self._fp
        self.offset = self.__mpoffsets[frame]
        self.fp.seek(self.offset + 2)
        segment = self.fp.read(2)
        if not segment:
            msg = 'No data found for frame'
            raise ValueError(msg)
        self._size = self._initial_size
        if i16(segment) == 65505:
            n = i16(self.fp.read(2)) - 2
            self.info['exif'] = ImageFile._safe_read(self.fp, n)
            self._reload_exif()
            mptype = self.mpinfo[45058][frame]['Attribute']['MPType']
            if mptype.startswith('Large Thumbnail'):
                exif = self.getexif().get_ifd(ExifTags.IFD.Exif)
                if 40962 in exif and 40963 in exif:
                    self._size = (exif[40962], exif[40963])
        elif 'exif' in self.info:
            del self.info['exif']
            self._reload_exif()
        self.tile = [('jpeg', (0, 0) + self.size, self.offset, (self.mode, ''))]
        self.__frame = frame

    def tell(self):
        return self.__frame

    @staticmethod
    def adopt(jpeg_instance, mpheader=None):
        """
        Transform the instance of JpegImageFile into
        an instance of MpoImageFile.
        After the call, the JpegImageFile is extended
        to be an MpoImageFile.

        This is essentially useful when opening a JPEG
        file that reveals itself as an MPO, to avoid
        double call to _open.
        """
        jpeg_instance.__class__ = MpoImageFile
        jpeg_instance._after_jpeg_open(mpheader)
        return jpeg_instance