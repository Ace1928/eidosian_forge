import numpy as np
from ..core import Format
class NpzFormat(Format):
    """See :mod:`imageio.plugins.npz`"""

    def _can_read(self, request):
        return request.extension in self.extensions

    def _can_write(self, request):
        return request.extension in self.extensions

    class Reader(Format.Reader):

        def _open(self):
            self._npz = np.load(self.request.get_file())
            assert isinstance(self._npz, np.lib.npyio.NpzFile)
            self._names = sorted(self._npz.files, key=lambda x: x.split('_')[-1])

        def _close(self):
            self._npz.close()

        def _get_length(self):
            return len(self._names)

        def _get_data(self, index):
            if index < 0 or index >= len(self._names):
                raise IndexError('Index out of range while reading from nzp')
            im = self._npz[self._names[index]]
            return (im, {})

        def _get_meta_data(self, index):
            raise RuntimeError('The npz format does not support meta data.')

    class Writer(Format.Writer):

        def _open(self):
            self._images = []

        def _close(self):
            np.savez_compressed(self.request.get_file(), *self._images)

        def _append_data(self, im, meta):
            self._images.append(im)

        def set_meta_data(self, meta):
            raise RuntimeError('The npz format does not support meta data.')