from warnings import warn
from ..auto import tqdm as tqdm_auto
from ..std import TqdmDeprecationWarning, tqdm
from ..utils import ObjectWrapper
class DummyTqdmFile(ObjectWrapper):
    """Dummy file-like that will write to tqdm"""

    def __init__(self, wrapped):
        super(DummyTqdmFile, self).__init__(wrapped)
        self._buf = []

    def write(self, x, nolock=False):
        nl = b'\n' if isinstance(x, bytes) else '\n'
        pre, sep, post = x.rpartition(nl)
        if sep:
            blank = type(nl)()
            tqdm.write(blank.join(self._buf + [pre, sep]), end=blank, file=self._wrapped, nolock=nolock)
            self._buf = [post]
        else:
            self._buf.append(x)

    def __del__(self):
        if self._buf:
            blank = type(self._buf[0])()
            try:
                tqdm.write(blank.join(self._buf), end=blank, file=self._wrapped)
            except (OSError, ValueError):
                pass