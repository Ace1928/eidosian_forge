import io
from typing import Any, Iterable, List, Optional
from urllib.parse import urlencode
from multidict import MultiDict, MultiDictProxy
from . import hdrs, multipart, payload
from .helpers import guess_filename
from .payload import Payload
def add_fields(self, *fields: Any) -> None:
    to_add = list(fields)
    while to_add:
        rec = to_add.pop(0)
        if isinstance(rec, io.IOBase):
            k = guess_filename(rec, 'unknown')
            self.add_field(k, rec)
        elif isinstance(rec, (MultiDictProxy, MultiDict)):
            to_add.extend(rec.items())
        elif isinstance(rec, (list, tuple)) and len(rec) == 2:
            k, fp = rec
            self.add_field(k, fp)
        else:
            raise TypeError('Only io.IOBase, multidict and (name, file) pairs allowed, use .add_field() for passing more complex parameters, got {!r}'.format(rec))