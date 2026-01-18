from __future__ import annotations
import itertools
import logging
import re
import struct
import warnings
import zlib
from enum import IntEnum
from . import Image, ImageChops, ImageFile, ImagePalette, ImageSequence
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from ._binary import o32be as o32
def _seek(self, frame, rewind=False):
    if frame == 0:
        if rewind:
            self._fp.seek(self.__rewind)
            self.png.rewind()
            self.__prepare_idat = self.__rewind_idat
            self.im = None
            if self.pyaccess:
                self.pyaccess = None
            self.info = self.png.im_info
            self.tile = self.png.im_tile
            self.fp = self._fp
        self._prev_im = None
        self.dispose = None
        self.default_image = self.info.get('default_image', False)
        self.dispose_op = self.info.get('disposal')
        self.blend_op = self.info.get('blend')
        self.dispose_extent = self.info.get('bbox')
        self.__frame = 0
    else:
        if frame != self.__frame + 1:
            msg = f'cannot seek to frame {frame}'
            raise ValueError(msg)
        self.load()
        if self.dispose:
            self.im.paste(self.dispose, self.dispose_extent)
        self._prev_im = self.im.copy()
        self.fp = self._fp
        if self.__prepare_idat:
            ImageFile._safe_read(self.fp, self.__prepare_idat)
            self.__prepare_idat = 0
        frame_start = False
        while True:
            self.fp.read(4)
            try:
                cid, pos, length = self.png.read()
            except (struct.error, SyntaxError):
                break
            if cid == b'IEND':
                msg = 'No more images in APNG file'
                raise EOFError(msg)
            if cid == b'fcTL':
                if frame_start:
                    msg = 'APNG missing frame data'
                    raise SyntaxError(msg)
                frame_start = True
            try:
                self.png.call(cid, pos, length)
            except UnicodeDecodeError:
                break
            except EOFError:
                if cid == b'fdAT':
                    length -= 4
                    if frame_start:
                        self.__prepare_idat = length
                        break
                ImageFile._safe_read(self.fp, length)
            except AttributeError:
                logger.debug('%r %s %s (unknown)', cid, pos, length)
                ImageFile._safe_read(self.fp, length)
        self.__frame = frame
        self.tile = self.png.im_tile
        self.dispose_op = self.info.get('disposal')
        self.blend_op = self.info.get('blend')
        self.dispose_extent = self.info.get('bbox')
        if not self.tile:
            msg = 'image not found in APNG frame'
            raise EOFError(msg)
    if self._prev_im is None and self.dispose_op == Disposal.OP_PREVIOUS:
        self.dispose_op = Disposal.OP_BACKGROUND
    if self.dispose_op == Disposal.OP_PREVIOUS:
        self.dispose = self._prev_im.copy()
        self.dispose = self._crop(self.dispose, self.dispose_extent)
    elif self.dispose_op == Disposal.OP_BACKGROUND:
        self.dispose = Image.core.fill(self.mode, self.size)
        self.dispose = self._crop(self.dispose, self.dispose_extent)
    else:
        self.dispose = None