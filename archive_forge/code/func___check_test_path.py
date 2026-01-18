from __future__ import annotations
import os
from . import (
from ...util import (
@staticmethod
def __check_test_path(paths: list[str], messages: LayoutMessages) -> None:
    modern_test_path = 'tests/'
    modern_test_path_found = any((path.startswith(modern_test_path) for path in paths))
    legacy_test_path = 'test/'
    legacy_test_path_found = any((path.startswith(legacy_test_path) for path in paths))
    if modern_test_path_found and legacy_test_path_found:
        messages.warning.append('Ignoring tests in "%s" in favor of "%s".' % (legacy_test_path, modern_test_path))
    elif legacy_test_path_found:
        messages.warning.append('Ignoring tests in "%s" that should be in "%s".' % (legacy_test_path, modern_test_path))