from typing import Any, cast, List, Optional, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class RunnerAll(RESTObject):
    _repr_attr = 'description'