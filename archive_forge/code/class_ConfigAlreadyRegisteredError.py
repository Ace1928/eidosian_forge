from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import (
class ConfigAlreadyRegisteredError(ConfigError):
    """Raised when a module tries to register a configuration option that
    already exists.

    Should not be raised too much really, only when developing new fontTools
    modules.
    """

    def __init__(self, name):
        super().__init__(f'Config option {name} is already registered.')