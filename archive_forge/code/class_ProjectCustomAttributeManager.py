from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import DeleteMixin, ObjectDeleteMixin, RetrieveMixin, SetMixin
class ProjectCustomAttributeManager(RetrieveMixin, SetMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/custom_attributes'
    _obj_cls = ProjectCustomAttribute
    _from_parent_attrs = {'project_id': 'id'}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectCustomAttribute:
        return cast(ProjectCustomAttribute, super().get(id=id, lazy=lazy, **kwargs))