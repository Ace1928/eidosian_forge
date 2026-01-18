from __future__ import annotations
import itertools
import math
import os
import subprocess
from enum import IntEnum
from . import (
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16
class GifImageFile(ImageFile.ImageFile):
    format = 'GIF'
    format_description = 'Compuserve GIF'
    _close_exclusive_fp_after_loading = False
    global_palette = None

    def data(self):
        s = self.fp.read(1)
        if s and s[0]:
            return self.fp.read(s[0])
        return None

    def _is_palette_needed(self, p):
        for i in range(0, len(p), 3):
            if not i // 3 == p[i] == p[i + 1] == p[i + 2]:
                return True
        return False

    def _open(self):
        s = self.fp.read(13)
        if not _accept(s):
            msg = 'not a GIF file'
            raise SyntaxError(msg)
        self.info['version'] = s[:6]
        self._size = (i16(s, 6), i16(s, 8))
        self.tile = []
        flags = s[10]
        bits = (flags & 7) + 1
        if flags & 128:
            self.info['background'] = s[11]
            p = self.fp.read(3 << bits)
            if self._is_palette_needed(p):
                p = ImagePalette.raw('RGB', p)
                self.global_palette = self.palette = p
        self._fp = self.fp
        self.__rewind = self.fp.tell()
        self._n_frames = None
        self._is_animated = None
        self._seek(0)

    @property
    def n_frames(self):
        if self._n_frames is None:
            current = self.tell()
            try:
                while True:
                    self._seek(self.tell() + 1, False)
            except EOFError:
                self._n_frames = self.tell() + 1
            self.seek(current)
        return self._n_frames

    @property
    def is_animated(self):
        if self._is_animated is None:
            if self._n_frames is not None:
                self._is_animated = self._n_frames != 1
            else:
                current = self.tell()
                if current:
                    self._is_animated = True
                else:
                    try:
                        self._seek(1, False)
                        self._is_animated = True
                    except EOFError:
                        self._is_animated = False
                    self.seek(current)
        return self._is_animated

    def seek(self, frame):
        if not self._seek_check(frame):
            return
        if frame < self.__frame:
            self.im = None
            self._seek(0)
        last_frame = self.__frame
        for f in range(self.__frame + 1, frame + 1):
            try:
                self._seek(f)
            except EOFError as e:
                self.seek(last_frame)
                msg = 'no more images in GIF file'
                raise EOFError(msg) from e

    def _seek(self, frame, update_image=True):
        if frame == 0:
            self.__offset = 0
            self.dispose = None
            self.__frame = -1
            self._fp.seek(self.__rewind)
            self.disposal_method = 0
            if 'comment' in self.info:
                del self.info['comment']
        elif self.tile and update_image:
            self.load()
        if frame != self.__frame + 1:
            msg = f'cannot seek to frame {frame}'
            raise ValueError(msg)
        self.fp = self._fp
        if self.__offset:
            self.fp.seek(self.__offset)
            while self.data():
                pass
            self.__offset = 0
        s = self.fp.read(1)
        if not s or s == b';':
            msg = 'no more images in GIF file'
            raise EOFError(msg)
        palette = None
        info = {}
        frame_transparency = None
        interlace = None
        frame_dispose_extent = None
        while True:
            if not s:
                s = self.fp.read(1)
            if not s or s == b';':
                break
            elif s == b'!':
                s = self.fp.read(1)
                block = self.data()
                if s[0] == 249:
                    flags = block[0]
                    if flags & 1:
                        frame_transparency = block[3]
                    info['duration'] = i16(block, 1) * 10
                    dispose_bits = 28 & flags
                    dispose_bits = dispose_bits >> 2
                    if dispose_bits:
                        self.disposal_method = dispose_bits
                elif s[0] == 254:
                    comment = b''
                    while block:
                        comment += block
                        block = self.data()
                    if 'comment' in info:
                        info['comment'] += b'\n' + comment
                    else:
                        info['comment'] = comment
                    s = None
                    continue
                elif s[0] == 255 and frame == 0:
                    info['extension'] = (block, self.fp.tell())
                    if block[:11] == b'NETSCAPE2.0':
                        block = self.data()
                        if len(block) >= 3 and block[0] == 1:
                            self.info['loop'] = i16(block, 1)
                while self.data():
                    pass
            elif s == b',':
                s = self.fp.read(9)
                x0, y0 = (i16(s, 0), i16(s, 2))
                x1, y1 = (x0 + i16(s, 4), y0 + i16(s, 6))
                if (x1 > self.size[0] or y1 > self.size[1]) and update_image:
                    self._size = (max(x1, self.size[0]), max(y1, self.size[1]))
                    Image._decompression_bomb_check(self._size)
                frame_dispose_extent = (x0, y0, x1, y1)
                flags = s[8]
                interlace = flags & 64 != 0
                if flags & 128:
                    bits = (flags & 7) + 1
                    p = self.fp.read(3 << bits)
                    if self._is_palette_needed(p):
                        palette = ImagePalette.raw('RGB', p)
                    else:
                        palette = False
                bits = self.fp.read(1)[0]
                self.__offset = self.fp.tell()
                break
            s = None
        if interlace is None:
            msg = 'image not found in GIF frame'
            raise EOFError(msg)
        self.__frame = frame
        if not update_image:
            return
        self.tile = []
        if self.dispose:
            self.im.paste(self.dispose, self.dispose_extent)
        self._frame_palette = palette if palette is not None else self.global_palette
        self._frame_transparency = frame_transparency
        if frame == 0:
            if self._frame_palette:
                if LOADING_STRATEGY == LoadingStrategy.RGB_ALWAYS:
                    self._mode = 'RGBA' if frame_transparency is not None else 'RGB'
                else:
                    self._mode = 'P'
            else:
                self._mode = 'L'
            if not palette and self.global_palette:
                from copy import copy
                palette = copy(self.global_palette)
            self.palette = palette
        elif self.mode == 'P':
            if LOADING_STRATEGY != LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY or palette:
                self.pyaccess = None
                if 'transparency' in self.info:
                    self.im.putpalettealpha(self.info['transparency'], 0)
                    self.im = self.im.convert('RGBA', Image.Dither.FLOYDSTEINBERG)
                    self._mode = 'RGBA'
                    del self.info['transparency']
                else:
                    self._mode = 'RGB'
                    self.im = self.im.convert('RGB', Image.Dither.FLOYDSTEINBERG)

        def _rgb(color):
            if self._frame_palette:
                if color * 3 + 3 > len(self._frame_palette.palette):
                    color = 0
                color = tuple(self._frame_palette.palette[color * 3:color * 3 + 3])
            else:
                color = (color, color, color)
            return color
        self.dispose_extent = frame_dispose_extent
        try:
            if self.disposal_method < 2:
                self.dispose = None
            elif self.disposal_method == 2:
                x0, y0, x1, y1 = self.dispose_extent
                dispose_size = (x1 - x0, y1 - y0)
                Image._decompression_bomb_check(dispose_size)
                dispose_mode = 'P'
                color = self.info.get('transparency', frame_transparency)
                if color is not None:
                    if self.mode in ('RGB', 'RGBA'):
                        dispose_mode = 'RGBA'
                        color = _rgb(color) + (0,)
                else:
                    color = self.info.get('background', 0)
                    if self.mode in ('RGB', 'RGBA'):
                        dispose_mode = 'RGB'
                        color = _rgb(color)
                self.dispose = Image.core.fill(dispose_mode, dispose_size, color)
            elif self.im is not None:
                self.dispose = self._crop(self.im, self.dispose_extent)
            elif frame_transparency is not None:
                x0, y0, x1, y1 = self.dispose_extent
                dispose_size = (x1 - x0, y1 - y0)
                Image._decompression_bomb_check(dispose_size)
                dispose_mode = 'P'
                color = frame_transparency
                if self.mode in ('RGB', 'RGBA'):
                    dispose_mode = 'RGBA'
                    color = _rgb(frame_transparency) + (0,)
                self.dispose = Image.core.fill(dispose_mode, dispose_size, color)
        except AttributeError:
            pass
        if interlace is not None:
            transparency = -1
            if frame_transparency is not None:
                if frame == 0:
                    if LOADING_STRATEGY != LoadingStrategy.RGB_ALWAYS:
                        self.info['transparency'] = frame_transparency
                elif self.mode not in ('RGB', 'RGBA'):
                    transparency = frame_transparency
            self.tile = [('gif', (x0, y0, x1, y1), self.__offset, (bits, interlace, transparency))]
        if info.get('comment'):
            self.info['comment'] = info['comment']
        for k in ['duration', 'extension']:
            if k in info:
                self.info[k] = info[k]
            elif k in self.info:
                del self.info[k]

    def load_prepare(self):
        temp_mode = 'P' if self._frame_palette else 'L'
        self._prev_im = None
        if self.__frame == 0:
            if self._frame_transparency is not None:
                self.im = Image.core.fill(temp_mode, self.size, self._frame_transparency)
        elif self.mode in ('RGB', 'RGBA'):
            self._prev_im = self.im
            if self._frame_palette:
                self.im = Image.core.fill('P', self.size, self._frame_transparency or 0)
                self.im.putpalette(*self._frame_palette.getdata())
            else:
                self.im = None
        self._mode = temp_mode
        self._frame_palette = None
        super().load_prepare()

    def load_end(self):
        if self.__frame == 0:
            if self.mode == 'P' and LOADING_STRATEGY == LoadingStrategy.RGB_ALWAYS:
                if self._frame_transparency is not None:
                    self.im.putpalettealpha(self._frame_transparency, 0)
                    self._mode = 'RGBA'
                else:
                    self._mode = 'RGB'
                self.im = self.im.convert(self.mode, Image.Dither.FLOYDSTEINBERG)
            return
        if not self._prev_im:
            return
        if self._frame_transparency is not None:
            self.im.putpalettealpha(self._frame_transparency, 0)
            frame_im = self.im.convert('RGBA')
        else:
            frame_im = self.im.convert('RGB')
        frame_im = self._crop(frame_im, self.dispose_extent)
        self.im = self._prev_im
        self._mode = self.im.mode
        if frame_im.mode == 'RGBA':
            self.im.paste(frame_im, self.dispose_extent, frame_im)
        else:
            self.im.paste(frame_im, self.dispose_extent)

    def tell(self):
        return self.__frame