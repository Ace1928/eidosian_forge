from __future__ import annotations
import re
import typing as t
from dataclasses import dataclass
from dataclasses import field
from .converters import ValidationError
from .exceptions import NoMatch
from .exceptions import RequestAliasRedirect
from .exceptions import RequestPath
from .rules import Rule
from .rules import RulePart
def _update_state(state: State) -> None:
    state.dynamic.sort(key=lambda entry: entry[0].weight)
    for new_state in state.static.values():
        _update_state(new_state)
    for _, new_state in state.dynamic:
        _update_state(new_state)