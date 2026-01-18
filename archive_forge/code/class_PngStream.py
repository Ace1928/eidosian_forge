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
class PngStream(ChunkStream):

    def __init__(self, fp):
        super().__init__(fp)
        self.im_info = {}
        self.im_text = {}
        self.im_size = (0, 0)
        self.im_mode = None
        self.im_tile = None
        self.im_palette = None
        self.im_custom_mimetype = None
        self.im_n_frames = None
        self._seq_num = None
        self.rewind_state = None
        self.text_memory = 0

    def check_text_memory(self, chunklen):
        self.text_memory += chunklen
        if self.text_memory > MAX_TEXT_MEMORY:
            msg = f'Too much memory used in text chunks: {self.text_memory}>MAX_TEXT_MEMORY'
            raise ValueError(msg)

    def save_rewind(self):
        self.rewind_state = {'info': self.im_info.copy(), 'tile': self.im_tile, 'seq_num': self._seq_num}

    def rewind(self):
        self.im_info = self.rewind_state['info']
        self.im_tile = self.rewind_state['tile']
        self._seq_num = self.rewind_state['seq_num']

    def chunk_iCCP(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        i = s.find(b'\x00')
        logger.debug('iCCP profile name %r', s[:i])
        logger.debug('Compression method %s', s[i])
        comp_method = s[i]
        if comp_method != 0:
            msg = f'Unknown compression method {comp_method} in iCCP chunk'
            raise SyntaxError(msg)
        try:
            icc_profile = _safe_zlib_decompress(s[i + 2:])
        except ValueError:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                icc_profile = None
            else:
                raise
        except zlib.error:
            icc_profile = None
        self.im_info['icc_profile'] = icc_profile
        return s

    def chunk_IHDR(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        if length < 13:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                return s
            msg = 'Truncated IHDR chunk'
            raise ValueError(msg)
        self.im_size = (i32(s, 0), i32(s, 4))
        try:
            self.im_mode, self.im_rawmode = _MODES[s[8], s[9]]
        except Exception:
            pass
        if s[12]:
            self.im_info['interlace'] = 1
        if s[11]:
            msg = 'unknown filter category'
            raise SyntaxError(msg)
        return s

    def chunk_IDAT(self, pos, length):
        if 'bbox' in self.im_info:
            tile = [('zip', self.im_info['bbox'], pos, self.im_rawmode)]
        else:
            if self.im_n_frames is not None:
                self.im_info['default_image'] = True
            tile = [('zip', (0, 0) + self.im_size, pos, self.im_rawmode)]
        self.im_tile = tile
        self.im_idat = length
        msg = 'image data found'
        raise EOFError(msg)

    def chunk_IEND(self, pos, length):
        msg = 'end of PNG image'
        raise EOFError(msg)

    def chunk_PLTE(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        if self.im_mode == 'P':
            self.im_palette = ('RGB', s)
        return s

    def chunk_tRNS(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        if self.im_mode == 'P':
            if _simple_palette.match(s):
                i = s.find(b'\x00')
                if i >= 0:
                    self.im_info['transparency'] = i
            else:
                self.im_info['transparency'] = s
        elif self.im_mode in ('1', 'L', 'I'):
            self.im_info['transparency'] = i16(s)
        elif self.im_mode == 'RGB':
            self.im_info['transparency'] = (i16(s), i16(s, 2), i16(s, 4))
        return s

    def chunk_gAMA(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        self.im_info['gamma'] = i32(s) / 100000.0
        return s

    def chunk_cHRM(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        raw_vals = struct.unpack('>%dI' % (len(s) // 4), s)
        self.im_info['chromaticity'] = tuple((elt / 100000.0 for elt in raw_vals))
        return s

    def chunk_sRGB(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        if length < 1:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                return s
            msg = 'Truncated sRGB chunk'
            raise ValueError(msg)
        self.im_info['srgb'] = s[0]
        return s

    def chunk_pHYs(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        if length < 9:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                return s
            msg = 'Truncated pHYs chunk'
            raise ValueError(msg)
        px, py = (i32(s, 0), i32(s, 4))
        unit = s[8]
        if unit == 1:
            dpi = (px * 0.0254, py * 0.0254)
            self.im_info['dpi'] = dpi
        elif unit == 0:
            self.im_info['aspect'] = (px, py)
        return s

    def chunk_tEXt(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        try:
            k, v = s.split(b'\x00', 1)
        except ValueError:
            k = s
            v = b''
        if k:
            k = k.decode('latin-1', 'strict')
            v_str = v.decode('latin-1', 'replace')
            self.im_info[k] = v if k == 'exif' else v_str
            self.im_text[k] = v_str
            self.check_text_memory(len(v_str))
        return s

    def chunk_zTXt(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        try:
            k, v = s.split(b'\x00', 1)
        except ValueError:
            k = s
            v = b''
        if v:
            comp_method = v[0]
        else:
            comp_method = 0
        if comp_method != 0:
            msg = f'Unknown compression method {comp_method} in zTXt chunk'
            raise SyntaxError(msg)
        try:
            v = _safe_zlib_decompress(v[1:])
        except ValueError:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                v = b''
            else:
                raise
        except zlib.error:
            v = b''
        if k:
            k = k.decode('latin-1', 'strict')
            v = v.decode('latin-1', 'replace')
            self.im_info[k] = self.im_text[k] = v
            self.check_text_memory(len(v))
        return s

    def chunk_iTXt(self, pos, length):
        r = s = ImageFile._safe_read(self.fp, length)
        try:
            k, r = r.split(b'\x00', 1)
        except ValueError:
            return s
        if len(r) < 2:
            return s
        cf, cm, r = (r[0], r[1], r[2:])
        try:
            lang, tk, v = r.split(b'\x00', 2)
        except ValueError:
            return s
        if cf != 0:
            if cm == 0:
                try:
                    v = _safe_zlib_decompress(v)
                except ValueError:
                    if ImageFile.LOAD_TRUNCATED_IMAGES:
                        return s
                    else:
                        raise
                except zlib.error:
                    return s
            else:
                return s
        try:
            k = k.decode('latin-1', 'strict')
            lang = lang.decode('utf-8', 'strict')
            tk = tk.decode('utf-8', 'strict')
            v = v.decode('utf-8', 'strict')
        except UnicodeError:
            return s
        self.im_info[k] = self.im_text[k] = iTXt(v, lang, tk)
        self.check_text_memory(len(v))
        return s

    def chunk_eXIf(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        self.im_info['exif'] = b'Exif\x00\x00' + s
        return s

    def chunk_acTL(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        if length < 8:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                return s
            msg = 'APNG contains truncated acTL chunk'
            raise ValueError(msg)
        if self.im_n_frames is not None:
            self.im_n_frames = None
            warnings.warn('Invalid APNG, will use default PNG image if possible')
            return s
        n_frames = i32(s)
        if n_frames == 0 or n_frames > 2147483648:
            warnings.warn('Invalid APNG, will use default PNG image if possible')
            return s
        self.im_n_frames = n_frames
        self.im_info['loop'] = i32(s, 4)
        self.im_custom_mimetype = 'image/apng'
        return s

    def chunk_fcTL(self, pos, length):
        s = ImageFile._safe_read(self.fp, length)
        if length < 26:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                return s
            msg = 'APNG contains truncated fcTL chunk'
            raise ValueError(msg)
        seq = i32(s)
        if self._seq_num is None and seq != 0 or (self._seq_num is not None and self._seq_num != seq - 1):
            msg = 'APNG contains frame sequence errors'
            raise SyntaxError(msg)
        self._seq_num = seq
        width, height = (i32(s, 4), i32(s, 8))
        px, py = (i32(s, 12), i32(s, 16))
        im_w, im_h = self.im_size
        if px + width > im_w or py + height > im_h:
            msg = 'APNG contains invalid frames'
            raise SyntaxError(msg)
        self.im_info['bbox'] = (px, py, px + width, py + height)
        delay_num, delay_den = (i16(s, 20), i16(s, 22))
        if delay_den == 0:
            delay_den = 100
        self.im_info['duration'] = float(delay_num) / float(delay_den) * 1000
        self.im_info['disposal'] = s[24]
        self.im_info['blend'] = s[25]
        return s

    def chunk_fdAT(self, pos, length):
        if length < 4:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                s = ImageFile._safe_read(self.fp, length)
                return s
            msg = 'APNG contains truncated fDAT chunk'
            raise ValueError(msg)
        s = ImageFile._safe_read(self.fp, 4)
        seq = i32(s)
        if self._seq_num != seq - 1:
            msg = 'APNG contains frame sequence errors'
            raise SyntaxError(msg)
        self._seq_num = seq
        return self.chunk_IDAT(pos + 4, length - 4)