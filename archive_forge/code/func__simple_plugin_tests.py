from __future__ import annotations
import collections
import os
import re
import time
import typing as t
from ..target import (
from ..util import (
from .python import (
from .csharp import (
from .powershell import (
from ..config import (
from ..metadata import (
from ..data import (
def _simple_plugin_tests(self, plugin_type: str, plugin_name: str) -> dict[str, t.Optional[str]]:
    """
        Return tests for the given plugin type and plugin name.
        This function is useful for plugin types which do not require special processing.
        """
    if plugin_name == '__init__':
        return all_tests(self.args, True)
    integration_target = self.integration_targets_by_name.get('%s_%s' % (plugin_type, plugin_name))
    if integration_target:
        integration_name = integration_target.name
    else:
        integration_name = None
    units_path = os.path.join(data_context().content.unit_path, 'plugins', plugin_type, 'test_%s.py' % plugin_name)
    if units_path not in self.units_paths:
        units_path = None
    return dict(integration=integration_name, units=units_path)