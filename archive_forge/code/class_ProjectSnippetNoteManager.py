from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import (  # noqa: F401
class ProjectSnippetNoteManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/snippets/{snippet_id}/notes'
    _obj_cls = ProjectSnippetNote
    _from_parent_attrs = {'project_id': 'project_id', 'snippet_id': 'id'}
    _create_attrs = RequiredOptional(required=('body',))
    _update_attrs = RequiredOptional(required=('body',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectSnippetNote:
        return cast(ProjectSnippetNote, super().get(id=id, lazy=lazy, **kwargs))