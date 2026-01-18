import gzip
import os
from io import BytesIO
from .... import osutils
from ....errors import BzrError
from ....trace import mutter
from ....transport import FileExists, NoSuchFile
from . import TransportStore
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