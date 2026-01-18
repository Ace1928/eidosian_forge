from __future__ import annotations
import collections.abc as c
import os
from . import (
from ...constants import (
from ...test import (
from ...target import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...python_requirements import (
from ...config import (
from ...coverage_util import (
from ...data import (
from ...host_configs import (
from ...venv import (
def _get_module_test(module_restrictions: bool) -> c.Callable[[str], bool]:
    """Create a predicate which tests whether a path can be used by modules or not."""
    module_path = data_context().content.module_path
    module_utils_path = data_context().content.module_utils_path
    if module_restrictions:
        return lambda path: is_subdir(path, module_path) or is_subdir(path, module_utils_path)
    return lambda path: not (is_subdir(path, module_path) or is_subdir(path, module_utils_path))