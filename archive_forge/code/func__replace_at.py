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
def _replace_at(self, idx: int | tuple[int], new_key: Key | str, value: Item) -> None:
    value = _item(value)
    if isinstance(idx, tuple):
        for i in idx[1:]:
            self._body[i] = (None, Null())
        idx = idx[0]
    k, v = self._body[idx]
    if not isinstance(new_key, Key):
        if isinstance(value, (AoT, Table)) != isinstance(v, (AoT, Table)) or new_key != k.key:
            new_key = SingleKey(new_key)
        else:
            new_key = k
    del self._map[k]
    self._map[new_key] = idx
    if new_key != k:
        dict.__delitem__(self, k)
    if isinstance(value, (AoT, Table)) != isinstance(v, (AoT, Table)):
        self.remove(k)
        for i in range(idx, len(self._body)):
            if isinstance(self._body[i][1], (AoT, Table)):
                self._insert_at(i, new_key, value)
                idx = i
                break
        else:
            idx = -1
            self.append(new_key, value)
    else:
        if not isinstance(value, (Whitespace, AoT)):
            value.trivia.indent = v.trivia.indent
            value.trivia.comment_ws = value.trivia.comment_ws or v.trivia.comment_ws
            value.trivia.comment = value.trivia.comment or v.trivia.comment
            value.trivia.trail = v.trivia.trail
        self._body[idx] = (new_key, value)
    if hasattr(value, 'invalidate_display_name'):
        value.invalidate_display_name()
    if isinstance(value, Table):
        last, _ = self._previous_item_with_index()
        idx = last if idx < 0 else idx
        has_ws = ends_with_whitespace(value)
        next_ws = idx < last and isinstance(self._body[idx + 1][1], Whitespace)
        if idx < last and (not (next_ws or has_ws)):
            value.append(None, Whitespace('\n'))
        dict.__setitem__(self, new_key.key, value.value)