from typing import Any, cast, Dict, List, Optional, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
from .custom_attributes import UserCustomAttributeManager  # noqa: F401
from .events import UserEventManager  # noqa: F401
from .personal_access_tokens import UserPersonalAccessTokenManager  # noqa: F401
class CurrentUserRunnerManager(CreateMixin, RESTManager):
    _path = '/user/runners'
    _obj_cls = CurrentUserRunner
    _types = {'tag_list': types.CommaSeparatedListAttribute}
    _create_attrs = RequiredOptional(required=('runner_type',), optional=('group_id', 'project_id', 'description', 'paused', 'locked', 'run_untagged', 'tag_list', 'access_level', 'maximum_timeout', 'maintenance_note'))