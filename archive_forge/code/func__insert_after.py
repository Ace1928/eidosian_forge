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
def _insert_after(self, key: Key | str, other_key: Key | str, item: Any) -> Container:
    if key is None:
        raise ValueError('Key cannot be null in insert_after()')
    if key not in self:
        raise NonExistentKey(key)
    if not isinstance(key, Key):
        key = SingleKey(key)
    if not isinstance(other_key, Key):
        other_key = SingleKey(other_key)
    item = _item(item)
    idx = self._map[key]
    if isinstance(idx, tuple):
        idx = max(idx)
    current_item = self._body[idx][1]
    if '\n' not in current_item.trivia.trail:
        current_item.trivia.trail += '\n'
    for k, v in self._map.items():
        if isinstance(v, tuple):
            new_indices = []
            for v_ in v:
                if v_ > idx:
                    v_ = v_ + 1
                new_indices.append(v_)
            self._map[k] = tuple(new_indices)
        elif v > idx:
            self._map[k] = v + 1
    self._map[other_key] = idx + 1
    self._body.insert(idx + 1, (other_key, item))
    if key is not None:
        dict.__setitem__(self, other_key.key, item.value)
    return self