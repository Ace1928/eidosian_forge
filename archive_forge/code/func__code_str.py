from __future__ import annotations
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union
from .util import compat
from .util import preloaded as _preloaded
def _code_str(self) -> str:
    if not self.code:
        return ''
    else:
        return f'(Background on this {self._what_are_we} at: https://sqlalche.me/e/{_version_token}/{self.code})'