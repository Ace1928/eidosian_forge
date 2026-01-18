from __future__ import annotations
import abc
import os
import typing as t
from ..util import (
class ProviderNotFoundForPath(ApplicationError):
    """Exception generated when a path based provider cannot be found for a given path."""

    def __init__(self, provider_type: t.Type, path: str) -> None:
        super().__init__('No %s found for path: %s' % (provider_type.__name__, path))
        self.provider_type = provider_type
        self.path = path