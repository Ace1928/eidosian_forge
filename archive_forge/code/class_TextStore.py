import gzip
import os
from io import BytesIO
from .... import osutils
from ....errors import BzrError
from ....trace import mutter
from ....transport import FileExists, NoSuchFile
from . import TransportStore
class TextStore(TransportStore):
    """Store that holds files indexed by unique names.

    Files can be added, but not modified once they are in.  Typically
    the hash is used as the name, or something else known to be unique,
    such as a UUID.

    Files are stored uncompressed, with no delta compression.
    """

    def _add_compressed(self, fn, f):
        if isinstance(f, bytes):
            f = BytesIO(f)
        sio = BytesIO()
        gf = gzip.GzipFile(mode='wb', fileobj=sio)
        osutils.pumpfile(f, gf)
        gf.close()
        sio.seek(0)
        self._try_put(fn, sio)

    def _add(self, fn, f):
        if self._compressed:
            self._add_compressed(fn, f)
        else:
            self._try_put(fn, f)

    def _try_put(self, fn, f):
        try:
            self._transport.put_file(fn, f, mode=self._file_mode)
        except NoSuchFile:
            if not self._prefixed:
                raise
            try:
                self._transport.mkdir(os.path.dirname(fn), mode=self._dir_mode)
            except FileExists:
                pass
            self._transport.put_file(fn, f, mode=self._file_mode)

    def _get(self, fn):
        if fn.endswith('.gz'):
            return self._get_compressed(fn)
        else:
            return self._transport.get(fn)

    def _get_compressed(self, filename):
        """Returns a file reading from a particular entry."""
        f = self._transport.get(filename)
        if getattr(f, 'tell', None) is not None:
            return gzip.GzipFile(mode='rb', fileobj=f)
        try:
            sio = BytesIO(f.read())
            return gzip.GzipFile(mode='rb', fileobj=sio)
        finally:
            f.close()