from typing import Any, cast, Union
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class GroupDeployTokenManager(RetrieveMixin, CreateMixin, DeleteMixin, RESTManager):
    _path = '/groups/{group_id}/deploy_tokens'
    _from_parent_attrs = {'group_id': 'id'}
    _obj_cls = GroupDeployToken
    _create_attrs = RequiredOptional(required=('name', 'scopes'), optional=('expires_at', 'username'))
    _list_filters = ('scopes',)
    _types = {'scopes': types.ArrayAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupDeployToken:
        return cast(GroupDeployToken, super().get(id=id, lazy=lazy, **kwargs))