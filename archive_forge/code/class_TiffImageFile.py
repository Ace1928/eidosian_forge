from __future__ import annotations
import io
import itertools
import logging
import math
import os
import struct
import warnings
from collections.abc import MutableMapping
from fractions import Fraction
from numbers import Number, Rational
from . import ExifTags, Image, ImageFile, ImageOps, ImagePalette, TiffTags
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from .TiffTags import TYPES
class TiffImageFile(ImageFile.ImageFile):
    format = 'TIFF'
    format_description = 'Adobe TIFF'
    _close_exclusive_fp_after_loading = False

    def __init__(self, fp=None, filename=None):
        self.tag_v2 = None
        ' Image file directory (tag dictionary) '
        self.tag = None
        ' Legacy tag entries '
        super().__init__(fp, filename)

    def _open(self):
        """Open the first image in a TIFF file"""
        ifh = self.fp.read(8)
        if ifh[2] == 43:
            ifh += self.fp.read(8)
        self.tag_v2 = ImageFileDirectory_v2(ifh)
        self.ifd = None
        self.__first = self.__next = self.tag_v2.next
        self.__frame = -1
        self._fp = self.fp
        self._frame_pos = []
        self._n_frames = None
        logger.debug('*** TiffImageFile._open ***')
        logger.debug('- __first: %s', self.__first)
        logger.debug('- ifh: %s', repr(ifh))
        self._seek(0)

    @property
    def n_frames(self):
        if self._n_frames is None:
            current = self.tell()
            self._seek(len(self._frame_pos))
            while self._n_frames is None:
                self._seek(self.tell() + 1)
            self.seek(current)
        return self._n_frames

    def seek(self, frame):
        """Select a given frame as current image"""
        if not self._seek_check(frame):
            return
        self._seek(frame)
        Image._decompression_bomb_check(self.size)
        self.im = Image.core.new(self.mode, self.size)

    def _seek(self, frame):
        self.fp = self._fp
        self.fp.tell()
        while len(self._frame_pos) <= frame:
            if not self.__next:
                msg = 'no more images in TIFF file'
                raise EOFError(msg)
            logger.debug('Seeking to frame %s, on frame %s, __next %s, location: %s', frame, self.__frame, self.__next, self.fp.tell())
            self.fp.seek(self.__next)
            self._frame_pos.append(self.__next)
            logger.debug('Loading tags, location: %s', self.fp.tell())
            self.tag_v2.load(self.fp)
            if self.tag_v2.next in self._frame_pos:
                self.__next = 0
            else:
                self.__next = self.tag_v2.next
            if self.__next == 0:
                self._n_frames = frame + 1
            if len(self._frame_pos) == 1:
                self.is_animated = self.__next != 0
            self.__frame += 1
        self.fp.seek(self._frame_pos[frame])
        self.tag_v2.load(self.fp)
        self._reload_exif()
        self.tag = self.ifd = ImageFileDirectory_v1.from_v2(self.tag_v2)
        self.__frame = frame
        self._setup()

    def tell(self):
        """Return the current frame number"""
        return self.__frame

    def getxmp(self):
        """
        Returns a dictionary containing the XMP tags.
        Requires defusedxml to be installed.

        :returns: XMP tags in a dictionary.
        """
        return self._getxmp(self.tag_v2[XMP]) if XMP in self.tag_v2 else {}

    def get_photoshop_blocks(self):
        """
        Returns a dictionary of Photoshop "Image Resource Blocks".
        The keys are the image resource ID. For more information, see
        https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/#50577409_pgfId-1037727

        :returns: Photoshop "Image Resource Blocks" in a dictionary.
        """
        blocks = {}
        val = self.tag_v2.get(ExifTags.Base.ImageResources)
        if val:
            while val[:4] == b'8BIM':
                id = i16(val[4:6])
                n = math.ceil((val[6] + 1) / 2) * 2
                size = i32(val[6 + n:10 + n])
                data = val[10 + n:10 + n + size]
                blocks[id] = {'data': data}
                val = val[math.ceil((10 + n + size) / 2) * 2:]
        return blocks

    def load(self):
        if self.tile and self.use_load_libtiff:
            return self._load_libtiff()
        return super().load()

    def load_end(self):
        if not self.is_animated:
            self._close_exclusive_fp_after_loading = True
            self.fp.tell()
            exif = self.getexif()
            for key in TiffTags.TAGS_V2_GROUPS:
                if key not in exif:
                    continue
                exif.get_ifd(key)
        ImageOps.exif_transpose(self, in_place=True)
        if ExifTags.Base.Orientation in self.tag_v2:
            del self.tag_v2[ExifTags.Base.Orientation]

    def _load_libtiff(self):
        """Overload method triggered when we detect a compressed tiff
        Calls out to libtiff"""
        Image.Image.load(self)
        self.load_prepare()
        if not len(self.tile) == 1:
            msg = 'Not exactly one tile'
            raise OSError(msg)
        extents = self.tile[0][1]
        args = list(self.tile[0][3])
        try:
            fp = hasattr(self.fp, 'fileno') and self.fp.fileno()
            if hasattr(self.fp, 'flush'):
                self.fp.flush()
        except OSError:
            fp = False
        if fp:
            args[2] = fp
        decoder = Image._getdecoder(self.mode, 'libtiff', tuple(args), self.decoderconfig)
        try:
            decoder.setimage(self.im, extents)
        except ValueError as e:
            msg = "Couldn't set the image"
            raise OSError(msg) from e
        close_self_fp = self._exclusive_fp and (not self.is_animated)
        if hasattr(self.fp, 'getvalue'):
            logger.debug('have getvalue. just sending in a string from getvalue')
            n, err = decoder.decode(self.fp.getvalue())
        elif fp:
            logger.debug('have fileno, calling fileno version of the decoder.')
            if not close_self_fp:
                self.fp.seek(0)
            n, err = decoder.decode(b'fpfp')
        else:
            logger.debug("don't have fileno or getvalue. just reading")
            self.fp.seek(0)
            n, err = decoder.decode(self.fp.read())
        self.tile = []
        self.readonly = 0
        self.load_end()
        if close_self_fp:
            self.fp.close()
            self.fp = None
        if err < 0:
            raise OSError(err)
        return Image.Image.load(self)

    def _setup(self):
        """Setup this image object based on current tags"""
        if 48129 in self.tag_v2:
            msg = 'Windows Media Photo files not yet supported'
            raise OSError(msg)
        self._compression = COMPRESSION_INFO[self.tag_v2.get(COMPRESSION, 1)]
        self._planar_configuration = self.tag_v2.get(PLANAR_CONFIGURATION, 1)
        photo = self.tag_v2.get(PHOTOMETRIC_INTERPRETATION, 0)
        if self._compression == 'tiff_jpeg':
            photo = 6
        fillorder = self.tag_v2.get(FILLORDER, 1)
        logger.debug('*** Summary ***')
        logger.debug('- compression: %s', self._compression)
        logger.debug('- photometric_interpretation: %s', photo)
        logger.debug('- planar_configuration: %s', self._planar_configuration)
        logger.debug('- fill_order: %s', fillorder)
        logger.debug('- YCbCr subsampling: %s', self.tag.get(YCBCRSUBSAMPLING))
        xsize = int(self.tag_v2.get(IMAGEWIDTH))
        ysize = int(self.tag_v2.get(IMAGELENGTH))
        self._size = (xsize, ysize)
        logger.debug('- size: %s', self.size)
        sample_format = self.tag_v2.get(SAMPLEFORMAT, (1,))
        if len(sample_format) > 1 and max(sample_format) == min(sample_format) == 1:
            sample_format = (1,)
        bps_tuple = self.tag_v2.get(BITSPERSAMPLE, (1,))
        extra_tuple = self.tag_v2.get(EXTRASAMPLES, ())
        if photo in (2, 6, 8):
            bps_count = 3
        elif photo == 5:
            bps_count = 4
        else:
            bps_count = 1
        bps_count += len(extra_tuple)
        bps_actual_count = len(bps_tuple)
        samples_per_pixel = self.tag_v2.get(SAMPLESPERPIXEL, 3 if self._compression == 'tiff_jpeg' and photo in (2, 6) else 1)
        if samples_per_pixel > MAX_SAMPLESPERPIXEL:
            logger.error('More samples per pixel than can be decoded: %s', samples_per_pixel)
            msg = 'Invalid value for samples per pixel'
            raise SyntaxError(msg)
        if samples_per_pixel < bps_actual_count:
            bps_tuple = bps_tuple[:samples_per_pixel]
        elif samples_per_pixel > bps_actual_count and bps_actual_count == 1:
            bps_tuple = bps_tuple * samples_per_pixel
        if len(bps_tuple) != samples_per_pixel:
            msg = 'unknown data organization'
            raise SyntaxError(msg)
        key = (self.tag_v2.prefix, photo, sample_format, fillorder, bps_tuple, extra_tuple)
        logger.debug('format key: %s', key)
        try:
            self._mode, rawmode = OPEN_INFO[key]
        except KeyError as e:
            logger.debug('- unsupported format')
            msg = 'unknown pixel mode'
            raise SyntaxError(msg) from e
        logger.debug('- raw mode: %s', rawmode)
        logger.debug('- pil mode: %s', self.mode)
        self.info['compression'] = self._compression
        xres = self.tag_v2.get(X_RESOLUTION, 1)
        yres = self.tag_v2.get(Y_RESOLUTION, 1)
        if xres and yres:
            resunit = self.tag_v2.get(RESOLUTION_UNIT)
            if resunit == 2:
                self.info['dpi'] = (xres, yres)
            elif resunit == 3:
                self.info['dpi'] = (xres * 2.54, yres * 2.54)
            elif resunit is None:
                self.info['dpi'] = (xres, yres)
                self.info['resolution'] = (xres, yres)
            else:
                self.info['resolution'] = (xres, yres)
        x = y = layer = 0
        self.tile = []
        self.use_load_libtiff = READ_LIBTIFF or self._compression != 'raw'
        if self.use_load_libtiff:
            if fillorder == 2:
                key = key[:3] + (1,) + key[4:]
                logger.debug('format key: %s', key)
                self._mode, rawmode = OPEN_INFO[key]
            if rawmode == 'I;16':
                rawmode = 'I;16N'
            if ';16B' in rawmode:
                rawmode = rawmode.replace(';16B', ';16N')
            if ';16L' in rawmode:
                rawmode = rawmode.replace(';16L', ';16N')
            if photo == 6 and self._compression == 'jpeg' and (self._planar_configuration == 1):
                rawmode = 'RGB'
            a = (rawmode, self._compression, False, self.tag_v2.offset)
            self.tile.append(('libtiff', (0, 0, xsize, ysize), 0, a))
        elif STRIPOFFSETS in self.tag_v2 or TILEOFFSETS in self.tag_v2:
            if STRIPOFFSETS in self.tag_v2:
                offsets = self.tag_v2[STRIPOFFSETS]
                h = self.tag_v2.get(ROWSPERSTRIP, ysize)
                w = self.size[0]
            else:
                offsets = self.tag_v2[TILEOFFSETS]
                w = self.tag_v2.get(TILEWIDTH)
                h = self.tag_v2.get(TILELENGTH)
            for offset in offsets:
                if x + w > xsize:
                    stride = w * sum(bps_tuple) / 8
                else:
                    stride = 0
                tile_rawmode = rawmode
                if self._planar_configuration == 2:
                    tile_rawmode = rawmode[layer]
                    stride /= bps_count
                a = (tile_rawmode, int(stride), 1)
                self.tile.append((self._compression, (x, y, min(x + w, xsize), min(y + h, ysize)), offset, a))
                x = x + w
                if x >= self.size[0]:
                    x, y = (0, y + h)
                    if y >= self.size[1]:
                        x = y = 0
                        layer += 1
        else:
            logger.debug('- unsupported data organization')
            msg = 'unknown data organization'
            raise SyntaxError(msg)
        if ICCPROFILE in self.tag_v2:
            self.info['icc_profile'] = self.tag_v2[ICCPROFILE]
        if self.mode in ['P', 'PA']:
            palette = [o8(b // 256) for b in self.tag_v2[COLORMAP]]
            self.palette = ImagePalette.raw('RGB;L', b''.join(palette))