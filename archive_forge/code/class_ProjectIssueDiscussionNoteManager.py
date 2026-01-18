from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import (  # noqa: F401
class ProjectIssueDiscussionNoteManager(GetMixin, CreateMixin, UpdateMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/issues/{issue_iid}/discussions/{discussion_id}/notes'
    _obj_cls = ProjectIssueDiscussionNote
    _from_parent_attrs = {'project_id': 'project_id', 'issue_iid': 'issue_iid', 'discussion_id': 'id'}
    _create_attrs = RequiredOptional(required=('body',), optional=('created_at',))
    _update_attrs = RequiredOptional(required=('body',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectIssueDiscussionNote:
        return cast(ProjectIssueDiscussionNote, super().get(id=id, lazy=lazy, **kwargs))