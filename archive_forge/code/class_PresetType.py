from __future__ import annotations
from collections.abc import MutableMapping as MutableMappingABC
from pathlib import Path
from typing import Any, Callable, Iterable, MutableMapping, TypedDict, cast
class PresetType(TypedDict):
    """Preset configuration for markdown-it."""
    options: OptionsType
    'Options for parsing.'
    components: MutableMapping[str, MutableMapping[str, list[str]]]
    'Components for parsing and rendering.'