import gzip
import os
from io import BytesIO
from .... import osutils
from ....errors import BzrError
from ....trace import mutter
from ....transport import FileExists, NoSuchFile
from . import TransportStore
def _add_compressed(self, fn, f):
    if isinstance(f, bytes):
        f = BytesIO(f)
    sio = BytesIO()
    gf = gzip.GzipFile(mode='wb', fileobj=sio)
    osutils.pumpfile(f, gf)
    gf.close()
    sio.seek(0)
    self._try_put(fn, sio)