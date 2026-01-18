import logging
import numpy as np
from .pillow_legacy import PillowFormat, image_as_uint, ndarray_to_pil
class GifWriter:
    """Class that for helping write the animated GIF file. This is based on
    code from images2gif.py (part of visvis). The version here is modified
    to allow streamed writing.
    """

    def __init__(self, file, opt_subrectangle=True, opt_loop=0, opt_quantizer=0, opt_palette_size=256):
        self.fp = file
        self.opt_subrectangle = opt_subrectangle
        self.opt_loop = opt_loop
        self.opt_quantizer = opt_quantizer
        self.opt_palette_size = opt_palette_size
        self._previous_image = None
        self._global_palette = None
        self._count = 0
        from PIL.GifImagePlugin import getdata
        self.getdata = getdata

    def add_image(self, im, duration, dispose):
        im_rect, rect = (im, (0, 0))
        if self.opt_subrectangle:
            im_rect, rect = self.getSubRectangle(im)
        im_pil = self.converToPIL(im_rect, self.opt_quantizer, self.opt_palette_size)
        from PIL.GifImagePlugin import getheader
        palette = getheader(im_pil)[0][3]
        if self._count == 0:
            self.write_header(im_pil, palette, self.opt_loop)
            self._global_palette = palette
        self.write_image(im_pil, palette, rect, duration, dispose)
        self._previous_image = im
        self._count += 1

    def write_header(self, im, globalPalette, loop):
        header = self.getheaderAnim(im)
        appext = self.getAppExt(loop)
        self.fp.write(header)
        self.fp.write(globalPalette)
        self.fp.write(appext)

    def close(self):
        self.fp.write(';'.encode('utf-8'))

    def write_image(self, im, palette, rect, duration, dispose):
        fp = self.fp
        data = self.getdata(im)
        imdes = b''
        while data and len(imdes) < 11:
            imdes += data.pop(0)
        assert len(imdes) == 11
        lid = self.getImageDescriptor(im, rect)
        graphext = self.getGraphicsControlExt(duration, dispose)
        if palette != self._global_palette or dispose != 2:
            fp.write(graphext)
            fp.write(lid)
            fp.write(palette)
            fp.write(b'\x08')
        else:
            fp.write(graphext)
            fp.write(imdes)
        for d in data:
            fp.write(d)

    def getheaderAnim(self, im):
        """Get animation header. To replace PILs getheader()[0]"""
        bb = b'GIF89a'
        bb += intToBin(im.size[0])
        bb += intToBin(im.size[1])
        bb += b'\x87\x00\x00'
        return bb

    def getImageDescriptor(self, im, xy=None):
        """Used for the local color table properties per image.
        Otherwise global color table applies to all frames irrespective of
        whether additional colors comes in play that require a redefined
        palette. Still a maximum of 256 color per frame, obviously.

        Written by Ant1 on 2010-08-22
        Modified by Alex Robinson in Janurari 2011 to implement subrectangles.
        """
        if xy is None:
            xy = (0, 0)
        bb = b','
        bb += intToBin(xy[0])
        bb += intToBin(xy[1])
        bb += intToBin(im.size[0])
        bb += intToBin(im.size[1])
        bb += b'\x87'
        return bb

    def getAppExt(self, loop):
        """Application extension. This part specifies the amount of loops.
        If loop is 0 or inf, it goes on infinitely.
        """
        if loop == 1:
            return b''
        if loop == 0:
            loop = 2 ** 16 - 1
        bb = b''
        if loop != 0:
            bb = b'!\xff\x0b'
            bb += b'NETSCAPE2.0'
            bb += b'\x03\x01'
            bb += intToBin(loop)
            bb += b'\x00'
        return bb

    def getGraphicsControlExt(self, duration=0.1, dispose=2):
        """Graphics Control Extension. A sort of header at the start of
        each image. Specifies duration and transparancy.

        Dispose
        -------
          * 0 - No disposal specified.
          * 1 - Do not dispose. The graphic is to be left in place.
          * 2 - Restore to background color. The area used by the graphic
            must be restored to the background color.
          * 3 - Restore to previous. The decoder is required to restore the
            area overwritten by the graphic with what was there prior to
            rendering the graphic.
          * 4-7 -To be defined.
        """
        bb = b'!\xf9\x04'
        bb += chr((dispose & 3) << 2).encode('utf-8')
        bb += intToBin(int(duration * 100 + 0.5))
        bb += b'\x00'
        bb += b'\x00'
        return bb

    def getSubRectangle(self, im):
        """Calculate the minimal rectangle that need updating. Returns
        a two-element tuple containing the cropped image and an x-y tuple.

        Calculating the subrectangles takes extra time, obviously. However,
        if the image sizes were reduced, the actual writing of the GIF
        goes faster. In some cases applying this method produces a GIF faster.
        """
        if self._count == 0:
            return (im, (0, 0))
        prev = self._previous_image
        diff = np.abs(im - prev)
        if diff.ndim == 3:
            diff = diff.sum(2)
        X = np.argwhere(diff.sum(0))
        Y = np.argwhere(diff.sum(1))
        if X.size and Y.size:
            x0, x1 = (int(X[0]), int(X[-1] + 1))
            y0, y1 = (int(Y[0]), int(Y[-1] + 1))
        else:
            x0, x1 = (0, 2)
            y0, y1 = (0, 2)
        return (im[y0:y1, x0:x1], (x0, y0))

    def converToPIL(self, im, quantizer, palette_size=256):
        """Convert image to Paletted PIL image.

        PIL used to not do a very good job at quantization, but I guess
        this has improved a lot (at least in Pillow). I don't think we need
        neuqant (and we can add it later if we really want).
        """
        im_pil = ndarray_to_pil(im, 'gif')
        if quantizer in ('nq', 'neuquant'):
            nq_samplefac = 10
            im_pil = im_pil.convert('RGBA')
            nqInstance = NeuQuant(im_pil, nq_samplefac)
            im_pil = nqInstance.quantize(im_pil, colors=palette_size)
        elif quantizer in (0, 1, 2):
            if quantizer == 2:
                im_pil = im_pil.convert('RGBA')
            else:
                im_pil = im_pil.convert('RGB')
            im_pil = im_pil.quantize(colors=palette_size, method=quantizer)
        else:
            raise ValueError('Invalid value for quantizer: %r' % quantizer)
        return im_pil