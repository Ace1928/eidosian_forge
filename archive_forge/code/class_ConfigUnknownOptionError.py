from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import (
class ConfigUnknownOptionError(ConfigError):
    """Raised when a configuration option is unknown."""

    def __init__(self, option_or_name):
        name = f"'{option_or_name.name}' (id={id(option_or_name)})>" if isinstance(option_or_name, Option) else f"'{option_or_name}'"
        super().__init__(f'Config option {name} is unknown')