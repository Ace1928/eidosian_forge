from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import DeleteMixin, ObjectDeleteMixin, RetrieveMixin, SetMixin
class UserCustomAttributeManager(RetrieveMixin, SetMixin, DeleteMixin, RESTManager):
    _path = '/users/{user_id}/custom_attributes'
    _obj_cls = UserCustomAttribute
    _from_parent_attrs = {'user_id': 'id'}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> UserCustomAttribute:
        return cast(UserCustomAttribute, super().get(id=id, lazy=lazy, **kwargs))