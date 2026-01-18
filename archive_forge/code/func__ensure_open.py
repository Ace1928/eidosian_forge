from ._base import *
@timed_cache(15)
def _ensure_open(self):
    if self.is_closed:
        raise ValueError('File is closed.')