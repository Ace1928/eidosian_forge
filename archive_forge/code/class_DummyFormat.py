import numpy as np
from .. import formats
from ..core import Format
class DummyFormat(Format):
    """The dummy format is an example format that does nothing.
    It will never indicate that it can read or write a file. When
    explicitly asked to read, it will simply read the bytes. When
    explicitly asked to write, it will raise an error.

    This documentation is shown when the user does ``help('thisformat')``.

    Parameters for reading
    ----------------------
    Specify arguments in numpy doc style here.

    Parameters for saving
    ---------------------
    Specify arguments in numpy doc style here.

    """

    def _can_read(self, request):
        if request.extension in self.extensions:
            return True

    def _can_write(self, request):
        if request.extension in self.extensions:
            return True

    class Reader(Format.Reader):

        def _open(self, some_option=False, length=1):
            self._fp = self.request.get_file()
            self._length = length
            self._data = None

        def _close(self):
            pass

        def _get_length(self):
            return self._length

        def _get_data(self, index):
            if index >= self._length:
                raise IndexError('Image index %i > %i' % (index, self._length))
            if self._data is None:
                self._data = self._fp.read()
            im = np.frombuffer(self._data, 'uint8')
            im.shape = (len(im), 1)
            return (im, {})

        def _get_meta_data(self, index):
            return {}

    class Writer(Format.Writer):

        def _open(self, flags=0):
            self._fp = self.request.get_file()

        def _close(self):
            pass

        def _append_data(self, im, meta):
            raise RuntimeError('The dummy format cannot write image data.')

        def set_meta_data(self, meta):
            raise RuntimeError('The dummy format cannot write meta data.')