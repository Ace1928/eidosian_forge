from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class ProjectBoardManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/boards'
    _obj_cls = ProjectBoard
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('name',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectBoard:
        return cast(ProjectBoard, super().get(id=id, lazy=lazy, **kwargs))