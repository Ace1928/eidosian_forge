from typing import Any, cast, Dict, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
class ProjectProtectedEnvironment(ObjectDeleteMixin, RESTObject):
    _id_attr = 'name'
    _repr_attr = 'name'