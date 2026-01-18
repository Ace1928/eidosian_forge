from ..core import Format, has_module
class GdalFormat(Format):
    """See :mod:`imageio.plugins.gdal`"""

    def _can_read(self, request):
        if request.extension in ('.ecw',):
            return True
        if has_module('osgeo.gdal'):
            return request.extension in self.extensions

    def _can_write(self, request):
        return False

    class Reader(Format.Reader):

        def _open(self):
            if not _gdal:
                load_lib()
            self._ds = _gdal.Open(self.request.get_local_filename())

        def _close(self):
            del self._ds

        def _get_length(self):
            return 1

        def _get_data(self, index):
            if index != 0:
                raise IndexError('Gdal file contains only one dataset')
            return (self._ds.ReadAsArray(), self._get_meta_data(index))

        def _get_meta_data(self, index):
            return self._ds.GetMetadata()