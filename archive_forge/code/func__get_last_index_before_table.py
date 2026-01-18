from __future__ import annotations
import copy
from typing import Any
from typing import Iterator
from tomlkit._compat import decode
from tomlkit._types import _CustomDict
from tomlkit._utils import merge_dicts
from tomlkit.exceptions import KeyAlreadyPresent
from tomlkit.exceptions import NonExistentKey
from tomlkit.exceptions import TOMLKitError
from tomlkit.items import AoT
from tomlkit.items import Comment
from tomlkit.items import Item
from tomlkit.items import Key
from tomlkit.items import Null
from tomlkit.items import SingleKey
from tomlkit.items import Table
from tomlkit.items import Trivia
from tomlkit.items import Whitespace
from tomlkit.items import item as _item
def _get_last_index_before_table(self) -> int:
    last_index = -1
    for i, (k, v) in enumerate(self._body):
        if isinstance(v, Null):
            continue
        if isinstance(v, Whitespace) and (not v.is_fixed()):
            continue
        if isinstance(v, (Table, AoT)) and (not k.is_dotted()):
            break
        last_index = i
    return last_index + 1