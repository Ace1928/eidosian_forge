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
class PngImageFile(ImageFile.ImageFile):
    format = 'PNG'
    format_description = 'Portable network graphics'

    def _open(self):
        if not _accept(self.fp.read(8)):
            msg = 'not a PNG file'
            raise SyntaxError(msg)
        self._fp = self.fp
        self.__frame = 0
        self.private_chunks = []
        self.png = PngStream(self.fp)
        while True:
            cid, pos, length = self.png.read()
            try:
                s = self.png.call(cid, pos, length)
            except EOFError:
                break
            except AttributeError:
                logger.debug('%r %s %s (unknown)', cid, pos, length)
                s = ImageFile._safe_read(self.fp, length)
                if cid[1:2].islower():
                    self.private_chunks.append((cid, s))
            self.png.crc(cid, s)
        self._mode = self.png.im_mode
        self._size = self.png.im_size
        self.info = self.png.im_info
        self._text = None
        self.tile = self.png.im_tile
        self.custom_mimetype = self.png.im_custom_mimetype
        self.n_frames = self.png.im_n_frames or 1
        self.default_image = self.info.get('default_image', False)
        if self.png.im_palette:
            rawmode, data = self.png.im_palette
            self.palette = ImagePalette.raw(rawmode, data)
        if cid == b'fdAT':
            self.__prepare_idat = length - 4
        else:
            self.__prepare_idat = length
        if self.png.im_n_frames is not None:
            self._close_exclusive_fp_after_loading = False
            self.png.save_rewind()
            self.__rewind_idat = self.__prepare_idat
            self.__rewind = self._fp.tell()
            if self.default_image:
                self.n_frames += 1
            self._seek(0)
        self.is_animated = self.n_frames > 1

    @property
    def text(self):
        if self._text is None:
            if self.is_animated:
                frame = self.__frame
                self.seek(self.n_frames - 1)
            self.load()
            if self.is_animated:
                self.seek(frame)
        return self._text

    def verify(self):
        """Verify PNG file"""
        if self.fp is None:
            msg = 'verify must be called directly after open'
            raise RuntimeError(msg)
        self.fp.seek(self.tile[0][2] - 8)
        self.png.verify()
        self.png.close()
        if self._exclusive_fp:
            self.fp.close()
        self.fp = None

    def seek(self, frame):
        if not self._seek_check(frame):
            return
        if frame < self.__frame:
            self._seek(0, True)
        last_frame = self.__frame
        for f in range(self.__frame + 1, frame + 1):
            try:
                self._seek(f)
            except EOFError as e:
                self.seek(last_frame)
                msg = 'no more images in APNG file'
                raise EOFError(msg) from e

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

    def tell(self):
        return self.__frame

    def load_prepare(self):
        """internal: prepare to read PNG file"""
        if self.info.get('interlace'):
            self.decoderconfig = self.decoderconfig + (1,)
        self.__idat = self.__prepare_idat
        ImageFile.ImageFile.load_prepare(self)

    def load_read(self, read_bytes):
        """internal: read more image data"""
        while self.__idat == 0:
            self.fp.read(4)
            cid, pos, length = self.png.read()
            if cid not in [b'IDAT', b'DDAT', b'fdAT']:
                self.png.push(cid, pos, length)
                return b''
            if cid == b'fdAT':
                try:
                    self.png.call(cid, pos, length)
                except EOFError:
                    pass
                self.__idat = length - 4
            else:
                self.__idat = length
        if read_bytes <= 0:
            read_bytes = self.__idat
        else:
            read_bytes = min(read_bytes, self.__idat)
        self.__idat = self.__idat - read_bytes
        return self.fp.read(read_bytes)

    def load_end(self):
        """internal: finished reading image data"""
        if self.__idat != 0:
            self.fp.read(self.__idat)
        while True:
            self.fp.read(4)
            try:
                cid, pos, length = self.png.read()
            except (struct.error, SyntaxError):
                break
            if cid == b'IEND':
                break
            elif cid == b'fcTL' and self.is_animated:
                self.__prepare_idat = 0
                self.png.push(cid, pos, length)
                break
            try:
                self.png.call(cid, pos, length)
            except UnicodeDecodeError:
                break
            except EOFError:
                if cid == b'fdAT':
                    length -= 4
                ImageFile._safe_read(self.fp, length)
            except AttributeError:
                logger.debug('%r %s %s (unknown)', cid, pos, length)
                s = ImageFile._safe_read(self.fp, length)
                if cid[1:2].islower():
                    self.private_chunks.append((cid, s, True))
        self._text = self.png.im_text
        if not self.is_animated:
            self.png.close()
            self.png = None
        elif self._prev_im and self.blend_op == Blend.OP_OVER:
            updated = self._crop(self.im, self.dispose_extent)
            if self.im.mode == 'RGB' and 'transparency' in self.info:
                mask = updated.convert_transparent('RGBA', self.info['transparency'])
            else:
                mask = updated.convert('RGBA')
            self._prev_im.paste(updated, self.dispose_extent, mask)
            self.im = self._prev_im
            if self.pyaccess:
                self.pyaccess = None

    def _getexif(self):
        if 'exif' not in self.info:
            self.load()
        if 'exif' not in self.info and 'Raw profile type exif' not in self.info:
            return None
        return self.getexif()._get_merged_dict()

    def getexif(self):
        if 'exif' not in self.info:
            self.load()
        return super().getexif()

    def getxmp(self):
        """
        Returns a dictionary containing the XMP tags.
        Requires defusedxml to be installed.

        :returns: XMP tags in a dictionary.
        """
        return self._getxmp(self.info['XML:com.adobe.xmp']) if 'XML:com.adobe.xmp' in self.info else {}