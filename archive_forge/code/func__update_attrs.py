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
def _update_attrs(self, new_attrs: Dict[str, Any]) -> None:
    self.__dict__['_updated_attrs'] = {}
    self.__dict__['_attrs'] = new_attrs