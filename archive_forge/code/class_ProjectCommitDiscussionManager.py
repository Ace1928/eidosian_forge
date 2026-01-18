from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, RetrieveMixin, SaveMixin, UpdateMixin
from gitlab.types import RequiredOptional
from .notes import (  # noqa: F401
class ProjectCommitDiscussionManager(RetrieveMixin, CreateMixin, RESTManager):
    _path = '/projects/{project_id}/repository/commits/{commit_id}/discussions'
    _obj_cls = ProjectCommitDiscussion
    _from_parent_attrs = {'project_id': 'project_id', 'commit_id': 'id'}
    _create_attrs = RequiredOptional(required=('body',), optional=('created_at',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectCommitDiscussion:
        return cast(ProjectCommitDiscussion, super().get(id=id, lazy=lazy, **kwargs))