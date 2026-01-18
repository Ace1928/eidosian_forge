from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import ProjectMergeRequestAwardEmojiManager  # noqa: F401
from .commits import ProjectCommit, ProjectCommitManager
from .discussions import ProjectMergeRequestDiscussionManager  # noqa: F401
from .draft_notes import ProjectMergeRequestDraftNoteManager
from .events import (  # noqa: F401
from .issues import ProjectIssue, ProjectIssueManager
from .merge_request_approvals import (  # noqa: F401
from .notes import ProjectMergeRequestNoteManager  # noqa: F401
from .pipelines import ProjectMergeRequestPipelineManager  # noqa: F401
from .reviewers import ProjectMergeRequestReviewerDetailManager
class GroupMergeRequestManager(ListMixin, RESTManager):
    _path = '/groups/{group_id}/merge_requests'
    _obj_cls = GroupMergeRequest
    _from_parent_attrs = {'group_id': 'id'}
    _list_filters = ('state', 'order_by', 'sort', 'milestone', 'view', 'labels', 'created_after', 'created_before', 'updated_after', 'updated_before', 'scope', 'author_id', 'assignee_id', 'approver_ids', 'approved_by_ids', 'my_reaction_emoji', 'source_branch', 'target_branch', 'search', 'wip')
    _types = {'approver_ids': types.ArrayAttribute, 'approved_by_ids': types.ArrayAttribute, 'labels': types.CommaSeparatedListAttribute}