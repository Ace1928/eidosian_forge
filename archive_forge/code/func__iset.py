from .core import Interface
from .file import File
from toolz import valmap
from .utils import frame, framesplit
def _iset(self, key, value, **kwargs):
    return self.partd.iset(key, frame(self.encode(value)), **kwargs)