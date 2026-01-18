from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import (  # noqa: F401
class ProjectIssueNoteManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/issues/{issue_iid}/notes'
    _obj_cls = ProjectIssueNote
    _from_parent_attrs = {'project_id': 'project_id', 'issue_iid': 'iid'}
    _create_attrs = RequiredOptional(required=('body',), optional=('created_at',))
    _update_attrs = RequiredOptional(required=('body',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectIssueNote:
        return cast(ProjectIssueNote, super().get(id=id, lazy=lazy, **kwargs))