from typing import Any, cast, Union
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class GroupMemberAllManager(RetrieveMixin, RESTManager):
    _path = '/groups/{group_id}/members/all'
    _obj_cls = GroupMemberAll
    _from_parent_attrs = {'group_id': 'id'}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupMemberAll:
        return cast(GroupMemberAll, super().get(id=id, lazy=lazy, **kwargs))