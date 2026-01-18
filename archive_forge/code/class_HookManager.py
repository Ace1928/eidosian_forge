from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, NoUpdateMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class HookManager(NoUpdateMixin, RESTManager):
    _path = '/hooks'
    _obj_cls = Hook
    _create_attrs = RequiredOptional(required=('url',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> Hook:
        return cast(Hook, super().get(id=id, lazy=lazy, **kwargs))