from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class ProjectBoardListManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/boards/{board_id}/lists'
    _obj_cls = ProjectBoardList
    _from_parent_attrs = {'project_id': 'project_id', 'board_id': 'id'}
    _create_attrs = RequiredOptional(exclusive=('label_id', 'assignee_id', 'milestone_id'))
    _update_attrs = RequiredOptional(required=('position',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectBoardList:
        return cast(ProjectBoardList, super().get(id=id, lazy=lazy, **kwargs))