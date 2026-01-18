from typing import Any, cast, Union
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectMemberManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/members'
    _obj_cls = ProjectMember
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('access_level', 'user_id'), optional=('expires_at', 'tasks_to_be_done'))
    _update_attrs = RequiredOptional(required=('access_level',), optional=('expires_at',))
    _types = {'user_ids': types.ArrayAttribute, 'tasks_to_be_dones': types.ArrayAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectMember:
        return cast(ProjectMember, super().get(id=id, lazy=lazy, **kwargs))