from typing import Any, cast, Union
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectMemberAllManager(RetrieveMixin, RESTManager):
    _path = '/projects/{project_id}/members/all'
    _obj_cls = ProjectMemberAll
    _from_parent_attrs = {'project_id': 'id'}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectMemberAll:
        return cast(ProjectMemberAll, super().get(id=id, lazy=lazy, **kwargs))