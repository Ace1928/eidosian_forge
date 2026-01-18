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
def _render_aot_table(self, table: Table, prefix: str | None=None) -> str:
    cur = ''
    _key = prefix or ''
    open_, close = ('[[', ']]')
    cur += f'{table.trivia.indent}{open_}{decode(_key)}{close}{table.trivia.comment_ws}{decode(table.trivia.comment)}{table.trivia.trail}'
    for k, v in table.value.body:
        if isinstance(v, Table):
            if v.is_super_table():
                if k.is_dotted():
                    cur += self._render_table(k, v)
                else:
                    cur += self._render_table(k, v, prefix=_key)
            else:
                cur += self._render_table(k, v, prefix=_key)
        elif isinstance(v, AoT):
            cur += self._render_aot(k, v, prefix=_key)
        else:
            cur += self._render_simple_item(k, v)
    return cur