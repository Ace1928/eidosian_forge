from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, RetrieveMixin, SaveMixin, UpdateMixin
from gitlab.types import RequiredOptional
from .notes import (  # noqa: F401
class ProjectIssueDiscussionManager(RetrieveMixin, CreateMixin, RESTManager):
    _path = '/projects/{project_id}/issues/{issue_iid}/discussions'
    _obj_cls = ProjectIssueDiscussion
    _from_parent_attrs = {'project_id': 'project_id', 'issue_iid': 'iid'}
    _create_attrs = RequiredOptional(required=('body',), optional=('created_at',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectIssueDiscussion:
        return cast(ProjectIssueDiscussion, super().get(id=id, lazy=lazy, **kwargs))