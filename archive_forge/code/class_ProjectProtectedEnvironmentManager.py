from typing import Any, cast, Dict, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
class ProjectProtectedEnvironmentManager(RetrieveMixin, CreateMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/protected_environments'
    _obj_cls = ProjectProtectedEnvironment
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('name', 'deploy_access_levels'), optional=('required_approval_count', 'approval_rules'))
    _types = {'deploy_access_levels': ArrayAttribute, 'approval_rules': ArrayAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectProtectedEnvironment:
        return cast(ProjectProtectedEnvironment, super().get(id=id, lazy=lazy, **kwargs))