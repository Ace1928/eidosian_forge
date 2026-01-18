from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import DeleteMixin, ObjectDeleteMixin, RetrieveMixin, SetMixin
class GroupCustomAttributeManager(RetrieveMixin, SetMixin, DeleteMixin, RESTManager):
    _path = '/groups/{group_id}/custom_attributes'
    _obj_cls = GroupCustomAttribute
    _from_parent_attrs = {'group_id': 'id'}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupCustomAttribute:
        return cast(GroupCustomAttribute, super().get(id=id, lazy=lazy, **kwargs))