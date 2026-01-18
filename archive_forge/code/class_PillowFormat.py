import logging
import threading
import numpy as np
from ..core import Format, image_as_uint
from ..core.request import URI_FILE, URI_BYTES
from .pillowmulti import GIFFormat, TIFFFormat  # noqa: E402, F401
class PillowFormat(Format):
    """
    Base format class for Pillow formats.
    """
    _pillow_imported = False
    _Image = None
    _modes = 'i'
    _description = ''

    def __init__(self, *args, plugin_id: str=None, **kwargs):
        super(PillowFormat, self).__init__(*args, **kwargs)
        self._lock = threading.RLock()
        self._plugin_id = plugin_id

    @property
    def plugin_id(self):
        """The PIL plugin id."""
        return self._plugin_id

    def _init_pillow(self):
        with self._lock:
            if not self._pillow_imported:
                self._pillow_imported = True
                import PIL
                if not hasattr(PIL, '__version__'):
                    raise ImportError('Imageio Pillow plugin requires Pillow, not PIL!')
                from PIL import Image
                self._Image = Image
            elif self._Image is None:
                raise RuntimeError('Imageio Pillow plugin requires Pillow lib.')
            Image = self._Image
        if self.plugin_id in ('PNG', 'JPEG', 'BMP', 'GIF', 'PPM'):
            Image.preinit()
        else:
            Image.init()
        return Image

    def _can_read(self, request):
        Image = self._init_pillow()
        if self.plugin_id in Image.OPEN:
            factory, accept = Image.OPEN[self.plugin_id]
            if accept:
                if request.firstbytes and accept(request.firstbytes):
                    return True

    def _can_write(self, request):
        Image = self._init_pillow()
        if request.extension in self.extensions or request._uri_type in [URI_FILE, URI_BYTES]:
            if self.plugin_id in Image.SAVE:
                return True

    class Reader(Format.Reader):

        def _open(self, pilmode=None, as_gray=False):
            Image = self.format._init_pillow()
            try:
                factory, accept = Image.OPEN[self.format.plugin_id]
            except KeyError:
                raise RuntimeError('Format %s cannot read images.' % self.format.name)
            self._fp = self._get_file()
            self._im = factory(self._fp, '')
            if hasattr(Image, '_decompression_bomb_check'):
                Image._decompression_bomb_check(self._im.size)
            if self._im.palette and self._im.palette.dirty:
                self._im.palette.rawmode_saved = self._im.palette.rawmode
            pil_try_read(self._im)
            self._kwargs = dict(as_gray=as_gray, is_gray=_palette_is_grayscale(self._im))
            if pilmode is not None:
                self._kwargs['mode'] = pilmode
            self._length = 1
            if hasattr(self._im, 'n_frames'):
                self._length = self._im.n_frames

        def _get_file(self):
            self._we_own_fp = False
            return self.request.get_file()

        def _close(self):
            save_pillow_close(self._im)
            if self._we_own_fp:
                self._fp.close()

        def _get_length(self):
            return self._length

        def _seek(self, index):
            try:
                self._im.seek(index)
            except EOFError:
                raise IndexError('Could not seek to index %i' % index)

        def _get_data(self, index):
            if index >= self._length:
                raise IndexError('Image index %i > %i' % (index, self._length))
            i = self._im.tell()
            if i > index:
                self._seek(index)
            else:
                while i < index:
                    i += 1
                    self._seek(i)
            if self._im.palette and self._im.palette.dirty:
                self._im.palette.rawmode_saved = self._im.palette.rawmode
            self._im.getdata()[0]
            im = pil_get_frame(self._im, **self._kwargs)
            return (im, self._im.info)

        def _get_meta_data(self, index):
            if not (index is None or index == 0):
                raise IndexError()
            return self._im.info

    class Writer(Format.Writer):

        def _open(self):
            Image = self.format._init_pillow()
            try:
                self._save_func = Image.SAVE[self.format.plugin_id]
            except KeyError:
                raise RuntimeError('Format %s cannot write images.' % self.format.name)
            self._fp = self.request.get_file()
            self._meta = {}
            self._written = False

        def _close(self):
            pass

        def _append_data(self, im, meta):
            if self._written:
                raise RuntimeError('Format %s only supports single images.' % self.format.name)
            if im.ndim == 3 and im.shape[-1] == 1:
                im = im[:, :, 0]
            self._written = True
            self._meta.update(meta)
            img = ndarray_to_pil(im, self.format.plugin_id, self._meta.pop('prefer_uint8', True))
            if 'bits' in self._meta:
                img = img.quantize()
            img.save(self._fp, format=self.format.plugin_id, **self._meta)
            save_pillow_close(img)

        def set_meta_data(self, meta):
            self._meta.update(meta)