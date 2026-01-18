from __future__ import annotations
import os
import os.path
import sys
from types import FrameType
from typing import Any, Iterable, Iterator
from coverage.exceptions import PluginError
from coverage.misc import isolate_module
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter
from coverage.types import (
def add_dynamic_context(self, plugin: CoveragePlugin) -> None:
    """Add a dynamic context plugin.

        `plugin` is an instance of a third-party plugin class.  It must
        implement the :meth:`CoveragePlugin.dynamic_context` method.

        """
    self._add_plugin(plugin, self.context_switchers)