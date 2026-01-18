from __future__ import annotations
import os
from . import (
from ...util import (
@staticmethod
def __check_integration_path(paths: list[str], messages: LayoutMessages) -> str:
    modern_integration_path = 'roles/test/'
    modern_integration_path_found = any((path.startswith(modern_integration_path) for path in paths))
    legacy_integration_path = 'tests/integration/targets/'
    legacy_integration_path_found = any((path.startswith(legacy_integration_path) for path in paths))
    if modern_integration_path_found and legacy_integration_path_found:
        messages.warning.append('Ignoring tests in "%s" in favor of "%s".' % (legacy_integration_path, modern_integration_path))
        integration_targets_path = modern_integration_path
    elif legacy_integration_path_found:
        messages.info.append('Falling back to tests in "%s" because "%s" was not found.' % (legacy_integration_path, modern_integration_path))
        integration_targets_path = legacy_integration_path
    elif modern_integration_path_found:
        messages.info.append('Loading tests from "%s".' % modern_integration_path)
        integration_targets_path = modern_integration_path
    else:
        messages.error.append('Cannot run integration tests without "%s" or "%s".' % (modern_integration_path, legacy_integration_path))
        integration_targets_path = modern_integration_path
    return integration_targets_path