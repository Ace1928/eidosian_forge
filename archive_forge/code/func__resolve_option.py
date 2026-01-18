from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import (
def _resolve_option(self, option_or_name: Union[Option, str]) -> Option:
    if isinstance(option_or_name, Option):
        option = option_or_name
        if not self.options.is_registered(option):
            raise ConfigUnknownOptionError(option)
        return option
    elif isinstance(option_or_name, str):
        name = option_or_name
        try:
            return self.options[name]
        except KeyError:
            raise ConfigUnknownOptionError(name)
    else:
        raise TypeError(f'expected Option or str, found {type(option_or_name).__name__}: {option_or_name!r}')