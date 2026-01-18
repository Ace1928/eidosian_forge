import logging
import os
from collections.abc import Mapping
from email.headerregistry import Address
from functools import partial, reduce
from inspect import cleandoc
from itertools import chain
from types import MappingProxyType
from typing import (
from ..errors import RemovedConfigError
from ..warnings import SetuptoolsWarning
def _apply_project_table(dist: 'Distribution', config: dict, root_dir: _Path):
    project_table = config.get('project', {}).copy()
    if not project_table:
        return
    _handle_missing_dynamic(dist, project_table)
    _unify_entry_points(project_table)
    for field, value in project_table.items():
        norm_key = json_compatible_key(field)
        corresp = PYPROJECT_CORRESPONDENCE.get(norm_key, norm_key)
        if callable(corresp):
            corresp(dist, value, root_dir)
        else:
            _set_config(dist, corresp, value)