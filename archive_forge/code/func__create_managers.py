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
def _create_managers(self) -> None:
    for attr, annotation in sorted(self.__annotations__.items()):
        if attr in ('manager',):
            continue
        if not isinstance(annotation, (type, str)):
            continue
        if isinstance(annotation, type):
            cls_name = annotation.__name__
        else:
            cls_name = annotation
        if cls_name == 'RESTManager' or not cls_name.endswith('Manager'):
            continue
        cls = getattr(self._module, cls_name)
        manager = cls(self.manager.gitlab, parent=self)
        self.__dict__[attr] = manager