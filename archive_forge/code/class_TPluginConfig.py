from __future__ import annotations
import os
import pathlib
from types import FrameType, ModuleType
from typing import (
class TPluginConfig(Protocol):
    """Something that can provide options to a plugin."""

    def get_plugin_options(self, plugin: str) -> TConfigSectionOut:
        """Get the options for a plugin."""