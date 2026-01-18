from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, RetrieveMixin, SaveMixin, UpdateMixin
from gitlab.types import RequiredOptional
from .notes import (  # noqa: F401
class ProjectSnippetDiscussionManager(RetrieveMixin, CreateMixin, RESTManager):
    _path = '/projects/{project_id}/snippets/{snippet_id}/discussions'
    _obj_cls = ProjectSnippetDiscussion
    _from_parent_attrs = {'project_id': 'project_id', 'snippet_id': 'id'}
    _create_attrs = RequiredOptional(required=('body',), optional=('created_at',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectSnippetDiscussion:
        return cast(ProjectSnippetDiscussion, super().get(id=id, lazy=lazy, **kwargs))