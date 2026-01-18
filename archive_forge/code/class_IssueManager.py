from typing import Any, cast, Dict, Optional, Tuple, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import ProjectIssueAwardEmojiManager  # noqa: F401
from .discussions import ProjectIssueDiscussionManager  # noqa: F401
from .events import (  # noqa: F401
from .notes import ProjectIssueNoteManager  # noqa: F401
class IssueManager(RetrieveMixin, RESTManager):
    _path = '/issues'
    _obj_cls = Issue
    _list_filters = ('state', 'labels', 'milestone', 'scope', 'author_id', 'iteration_id', 'assignee_id', 'my_reaction_emoji', 'iids', 'order_by', 'sort', 'search', 'created_after', 'created_before', 'updated_after', 'updated_before')
    _types = {'iids': types.ArrayAttribute, 'labels': types.CommaSeparatedListAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> Issue:
        return cast(Issue, super().get(id=id, lazy=lazy, **kwargs))