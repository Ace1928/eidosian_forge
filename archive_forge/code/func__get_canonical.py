from __future__ import annotations
from collections import namedtuple
import re
def _get_canonical(self) -> str:
    """Get the canonical output string."""
    if self.micro == 0:
        ver = f'{self.major}.{self.minor}'
    else:
        ver = f'{self.major}.{self.minor}.{self.micro}'
    if self._is_pre():
        ver += f'{REL_MAP[self.release]}{self.pre}'
    if self._is_post():
        ver += f'.post{self.post}'
    if self._is_dev():
        ver += f'.dev{self.dev}'
    return ver