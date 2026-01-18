from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class GroupBoardManager(CRUDMixin, RESTManager):
    _path = '/groups/{group_id}/boards'
    _obj_cls = GroupBoard
    _from_parent_attrs = {'group_id': 'id'}
    _create_attrs = RequiredOptional(required=('name',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupBoard:
        return cast(GroupBoard, super().get(id=id, lazy=lazy, **kwargs))