import numpy as np
from ..core import Format
class BsdfFormat(Format):
    """The BSDF format enables reading and writing of image data in the
    BSDF serialization format. This format allows storage of images, volumes,
    and series thereof. Data can be of any numeric data type, and can
    optionally be compressed. Each image/volume can have associated
    meta data, which can consist of any data type supported by BSDF.

    By default, image data is lazily loaded; the actual image data is
    not read until it is requested. This allows storing multiple images
    in a single file and still have fast access to individual images.
    Alternatively, a series of images can be read in streaming mode, reading
    images as they are read (e.g. from http).

    BSDF is a simple generic binary format. It is easy to extend and there
    are standard extension definitions for 2D and 3D image data.
    Read more at http://bsdf.io.

    Parameters for reading
    ----------------------
    random_access : bool
        Whether individual images in the file can be read in random order.
        Defaults to True for normal files, and to False when reading from HTTP.
        If False, the file is read in "streaming mode", allowing reading
        files as they are read, but without support for "rewinding".
        Note that setting this to True when reading from HTTP, the whole file
        is read upon opening it (since lazy loading is not possible over HTTP).

    Parameters for saving
    ---------------------
    compression : {0, 1, 2}
        Use ``0`` or "no" for no compression, ``1`` or "zlib" for Zlib
        compression (same as zip files and PNG), and ``2`` or "bz2" for Bz2
        compression (more compact but slower). Default 1 (zlib).
        Note that some BSDF implementations may not support compression
        (e.g. JavaScript).

    """

    def _can_read(self, request):
        if request.mode[1] in self.modes + '?':
            if request.firstbytes.startswith(b'BSDF'):
                return True

    def _can_write(self, request):
        if request.mode[1] in self.modes + '?':
            if request.extension in self.extensions:
                return True

    class Reader(Format.Reader):

        def _open(self, random_access=None):
            assert self.request.firstbytes[:4] == b'BSDF', 'Not a BSDF file'
            if not (self.request.firstbytes[6:15] == b'M\x07image2D' or self.request.firstbytes[6:15] == b'M\x07image3D' or self.request.firstbytes[6:7] == b'l'):
                pass
            options = {}
            if self.request.filename.startswith(('http://', 'https://')):
                ra = False if random_access is None else bool(random_access)
                options['lazy_blob'] = False
                options['load_streaming'] = not ra
            else:
                ra = True if random_access is None else bool(random_access)
                options['lazy_blob'] = ra
                options['load_streaming'] = not ra
            file = self.request.get_file()
            bsdf, self._serializer = get_bsdf_serializer(options)
            self._stream = self._serializer.load(file)
            if isinstance(self._stream, dict) and 'meta' in self._stream and ('array' in self._stream):
                self._stream = Image(self._stream['array'], self._stream['meta'])
            if not isinstance(self._stream, (Image, list, bsdf.ListStream)):
                raise RuntimeError('BSDF file does not look seem to have an image container.')

        def _close(self):
            pass

        def _get_length(self):
            if isinstance(self._stream, Image):
                return 1
            elif isinstance(self._stream, list):
                return len(self._stream)
            elif self._stream.count < 0:
                return np.inf
            return self._stream.count

        def _get_data(self, index):
            if index < 0 or index >= self.get_length():
                raise IndexError('Image index %i not in [0 %i].' % (index, self.get_length()))
            if isinstance(self._stream, Image):
                image_ob = self._stream
            elif isinstance(self._stream, list):
                image_ob = self._stream[index]
            else:
                if index < self._stream.index:
                    raise IndexError('BSDF file is being read in streaming mode, thus does not allow rewinding.')
                while index > self._stream.index:
                    self._stream.next()
                image_ob = self._stream.next()
            if isinstance(image_ob, dict) and 'meta' in image_ob and ('array' in image_ob):
                image_ob = Image(image_ob['array'], image_ob['meta'])
            if isinstance(image_ob, Image):
                return (image_ob.get_array(), image_ob.get_meta())
            else:
                r = repr(image_ob)
                r = r if len(r) < 200 else r[:197] + '...'
                raise RuntimeError('BSDF file contains non-image ' + r)

        def _get_meta_data(self, index):
            return {}

    class Writer(Format.Writer):

        def _open(self, compression=1):
            options = {'compression': compression}
            bsdf, self._serializer = get_bsdf_serializer(options)
            if self.request.mode[1] in 'iv':
                self._stream = None
                self._written = False
            else:
                file = self.request.get_file()
                self._stream = bsdf.ListStream()
                self._serializer.save(file, self._stream)

        def _close(self):
            if self._stream is not None:
                self._stream.close(False)

        def _append_data(self, im, meta):
            ndim = None
            if self.request.mode[1] in 'iI':
                ndim = 2
            elif self.request.mode[1] in 'vV':
                ndim = 3
            else:
                ndim = 3
                if im.ndim == 2 or (im.ndim == 3 and im.shape[-1] <= 4):
                    ndim = 2
            assert ndim in (2, 3)
            if ndim == 2:
                assert im.ndim == 2 or (im.ndim == 3 and im.shape[-1] <= 4)
            else:
                assert im.ndim == 3 or (im.ndim == 4 and im.shape[-1] <= 4)
            if ndim == 2:
                ob = Image2D(im, meta)
            else:
                ob = Image3D(im, meta)
            if self._stream is None:
                assert not self._written, 'Cannot write singleton image twice'
                self._written = True
                file = self.request.get_file()
                self._serializer.save(file, ob)
            else:
                self._stream.append(ob)

        def set_meta_data(self, meta):
            raise RuntimeError('The BSDF format only supports per-image meta data.')