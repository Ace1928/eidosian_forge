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
class SnippetManager(CRUDMixin, RESTManager):
    _path = '/snippets'
    _obj_cls = Snippet
    _create_attrs = RequiredOptional(required=('title',), exclusive=('files', 'file_name'), optional=('description', 'content', 'visibility'))
    _update_attrs = RequiredOptional(optional=('title', 'files', 'file_name', 'content', 'visibility', 'description'))

    @cli.register_custom_action('SnippetManager')
    def public(self, **kwargs: Any) -> Union[RESTObjectList, List[RESTObject]]:
        """List all the public snippets.

        Args:
            all: If True the returned object will be a list
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabListError: If the list could not be retrieved

        Returns:
            A generator for the snippets list
        """
        return self.list(path='/snippets/public', **kwargs)

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> Snippet:
        return cast(Snippet, super().get(id=id, lazy=lazy, **kwargs))