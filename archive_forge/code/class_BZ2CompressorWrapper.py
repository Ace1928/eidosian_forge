import io
import zlib
from joblib.backports import LooseVersion
class BZ2CompressorWrapper(CompressorWrapper):
    prefix = _BZ2_PREFIX
    extension = '.bz2'

    def __init__(self):
        if bz2 is not None:
            self.fileobj_factory = bz2.BZ2File
        else:
            self.fileobj_factory = None

    def _check_versions(self):
        if bz2 is None:
            raise ValueError('bz2 module is not compiled on your python standard library.')

    def compressor_file(self, fileobj, compresslevel=None):
        """Returns an instance of a compressor file object."""
        self._check_versions()
        if compresslevel is None:
            return self.fileobj_factory(fileobj, 'wb')
        else:
            return self.fileobj_factory(fileobj, 'wb', compresslevel=compresslevel)

    def decompressor_file(self, fileobj):
        """Returns an instance of a decompressor file object."""
        self._check_versions()
        fileobj = self.fileobj_factory(fileobj, 'rb')
        return fileobj