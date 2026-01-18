import sys as _sys
from keyword import iskeyword as _iskeyword
def _asdict(self):
    """Return a new dict which maps field names to their values."""
    out = _dict(_zip(self._fields, self))
    out.update(self.__dict__)
    return out