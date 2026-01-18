from typing import Any, Callable, cast, Iterator, List, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin, UserAgentDetailMixin
from gitlab.types import RequiredOptional
from .award_emojis import ProjectSnippetAwardEmojiManager  # noqa: F401
from .discussions import ProjectSnippetDiscussionManager  # noqa: F401
from .notes import ProjectSnippetNoteManager  # noqa: F401
class ProjectSnippetManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/snippets'
    _obj_cls = ProjectSnippet
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('title', 'visibility'), exclusive=('files', 'file_name'), optional=('description', 'content'))
    _update_attrs = RequiredOptional(optional=('title', 'files', 'file_name', 'content', 'visibility', 'description'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectSnippet:
        return cast(ProjectSnippet, super().get(id=id, lazy=lazy, **kwargs))