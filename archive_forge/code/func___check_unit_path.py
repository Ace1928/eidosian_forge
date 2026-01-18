from __future__ import annotations
import os
from . import (
from ...util import (
@staticmethod
def __check_unit_path(paths: list[str], messages: LayoutMessages) -> None:
    modern_unit_path = 'tests/unit/'
    modern_unit_path_found = any((path.startswith(modern_unit_path) for path in paths))
    legacy_unit_path = 'tests/units/'
    legacy_unit_path_found = any((path.startswith(legacy_unit_path) for path in paths))
    if modern_unit_path_found and legacy_unit_path_found:
        messages.warning.append('Ignoring tests in "%s" in favor of "%s".' % (legacy_unit_path, modern_unit_path))
    elif legacy_unit_path_found:
        messages.warning.append('Rename "%s" to "%s" to run unit tests.' % (legacy_unit_path, modern_unit_path))
    elif modern_unit_path_found:
        pass
    else:
        messages.error.append('Cannot run unit tests without "%s".' % modern_unit_path)