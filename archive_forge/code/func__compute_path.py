import copy
import importlib
import json
import pprint
import textwrap
from types import ModuleType
from typing import Any, Dict, Iterable, Optional, Type, TYPE_CHECKING, Union
import gitlab
from gitlab import types as g_types
from gitlab.exceptions import GitlabParsingError
from .client import Gitlab, GitlabList
def _compute_path(self, path: Optional[str]=None) -> Optional[str]:
    self._parent_attrs = {}
    if path is None:
        path = self._path
    if path is None:
        return None
    if self._parent is None or not self._from_parent_attrs:
        return path
    data: Dict[str, Optional[gitlab.utils.EncodedId]] = {}
    for self_attr, parent_attr in self._from_parent_attrs.items():
        if not hasattr(self._parent, parent_attr):
            data[self_attr] = None
            continue
        data[self_attr] = gitlab.utils.EncodedId(getattr(self._parent, parent_attr))
    self._parent_attrs = data
    return path.format(**data)