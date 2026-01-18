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
class MergeRequestManager(ListMixin, RESTManager):
    _path = '/merge_requests'
    _obj_cls = MergeRequest
    _list_filters = ('state', 'order_by', 'sort', 'milestone', 'view', 'labels', 'with_labels_details', 'with_merge_status_recheck', 'created_after', 'created_before', 'updated_after', 'updated_before', 'scope', 'author_id', 'author_username', 'assignee_id', 'approver_ids', 'approved_by_ids', 'reviewer_id', 'reviewer_username', 'my_reaction_emoji', 'source_branch', 'target_branch', 'search', 'in', 'wip', 'not', 'environment', 'deployed_before', 'deployed_after')
    _types = {'approver_ids': types.ArrayAttribute, 'approved_by_ids': types.ArrayAttribute, 'in': types.CommaSeparatedListAttribute, 'labels': types.CommaSeparatedListAttribute}