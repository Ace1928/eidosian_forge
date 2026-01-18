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
class ProjectIssueManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/issues'
    _obj_cls = ProjectIssue
    _from_parent_attrs = {'project_id': 'id'}
    _list_filters = ('iids', 'state', 'labels', 'milestone', 'scope', 'author_id', 'iteration_id', 'assignee_id', 'my_reaction_emoji', 'order_by', 'sort', 'search', 'created_after', 'created_before', 'updated_after', 'updated_before')
    _create_attrs = RequiredOptional(required=('title',), optional=('description', 'confidential', 'assignee_ids', 'assignee_id', 'milestone_id', 'labels', 'created_at', 'due_date', 'merge_request_to_resolve_discussions_of', 'discussion_to_resolve'))
    _update_attrs = RequiredOptional(optional=('title', 'description', 'confidential', 'assignee_ids', 'assignee_id', 'milestone_id', 'labels', 'state_event', 'updated_at', 'due_date', 'discussion_locked'))
    _types = {'iids': types.ArrayAttribute, 'labels': types.CommaSeparatedListAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectIssue:
        return cast(ProjectIssue, super().get(id=id, lazy=lazy, **kwargs))