from __future__ import annotations
import array
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from .JpegPresets import presets
class JpegImageFile(ImageFile.ImageFile):
    format = 'JPEG'
    format_description = 'JPEG (ISO 10918)'

    def _open(self):
        s = self.fp.read(3)
        if not _accept(s):
            msg = 'not a JPEG file'
            raise SyntaxError(msg)
        s = b'\xff'
        self.bits = self.layers = 0
        self.layer = []
        self.huffman_dc = {}
        self.huffman_ac = {}
        self.quantization = {}
        self.app = {}
        self.applist = []
        self.icclist = []
        while True:
            i = s[0]
            if i == 255:
                s = s + self.fp.read(1)
                i = i16(s)
            else:
                s = self.fp.read(1)
                continue
            if i in MARKER:
                name, description, handler = MARKER[i]
                if handler is not None:
                    handler(self, i)
                if i == 65498:
                    rawmode = self.mode
                    if self.mode == 'CMYK':
                        rawmode = 'CMYK;I'
                    self.tile = [('jpeg', (0, 0) + self.size, 0, (rawmode, ''))]
                    break
                s = self.fp.read(1)
            elif i in {0, 65535}:
                s = b'\xff'
            elif i == 65280:
                s = self.fp.read(1)
            else:
                msg = 'no marker found'
                raise SyntaxError(msg)

    def load_read(self, read_bytes):
        """
        internal: read more image data
        For premature EOF and LOAD_TRUNCATED_IMAGES adds EOI marker
        so libjpeg can finish decoding
        """
        s = self.fp.read(read_bytes)
        if not s and ImageFile.LOAD_TRUNCATED_IMAGES and (not hasattr(self, '_ended')):
            self._ended = True
            return b'\xff\xd9'
        return s

    def draft(self, mode, size):
        if len(self.tile) != 1:
            return
        if self.decoderconfig:
            return
        d, e, o, a = self.tile[0]
        scale = 1
        original_size = self.size
        if a[0] == 'RGB' and mode in ['L', 'YCbCr']:
            self._mode = mode
            a = (mode, '')
        if size:
            scale = min(self.size[0] // size[0], self.size[1] // size[1])
            for s in [8, 4, 2, 1]:
                if scale >= s:
                    break
            e = (e[0], e[1], (e[2] - e[0] + s - 1) // s + e[0], (e[3] - e[1] + s - 1) // s + e[1])
            self._size = ((self.size[0] + s - 1) // s, (self.size[1] + s - 1) // s)
            scale = s
        self.tile = [(d, e, o, a)]
        self.decoderconfig = (scale, 0)
        box = (0, 0, original_size[0] / scale, original_size[1] / scale)
        return (self.mode, box)

    def load_djpeg(self):
        f, path = tempfile.mkstemp()
        os.close(f)
        if os.path.exists(self.filename):
            subprocess.check_call(['djpeg', '-outfile', path, self.filename])
        else:
            try:
                os.unlink(path)
            except OSError:
                pass
            msg = 'Invalid Filename'
            raise ValueError(msg)
        try:
            with Image.open(path) as _im:
                _im.load()
                self.im = _im.im
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._mode = self.im.mode
        self._size = self.im.size
        self.tile = []

    def _getexif(self):
        return _getexif(self)

    def _getmp(self):
        return _getmp(self)

    def getxmp(self):
        """
        Returns a dictionary containing the XMP tags.
        Requires defusedxml to be installed.

        :returns: XMP tags in a dictionary.
        """
        for segment, content in self.applist:
            if segment == 'APP1':
                marker, xmp_tags = content.split(b'\x00')[:2]
                if marker == b'http://ns.adobe.com/xap/1.0/':
                    return self._getxmp(xmp_tags)
        return {}