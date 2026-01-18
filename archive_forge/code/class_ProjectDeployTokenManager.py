from typing import Any, cast, Union
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectDeployTokenManager(RetrieveMixin, CreateMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/deploy_tokens'
    _from_parent_attrs = {'project_id': 'id'}
    _obj_cls = ProjectDeployToken
    _create_attrs = RequiredOptional(required=('name', 'scopes'), optional=('expires_at', 'username'))
    _list_filters = ('scopes',)
    _types = {'scopes': types.ArrayAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectDeployToken:
        return cast(ProjectDeployToken, super().get(id=id, lazy=lazy, **kwargs))