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
class ProjectMergeRequestManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/merge_requests'
    _obj_cls = ProjectMergeRequest
    _from_parent_attrs = {'project_id': 'id'}
    _optional_get_attrs = ('render_html', 'include_diverged_commits_count', 'include_rebase_in_progress')
    _create_attrs = RequiredOptional(required=('source_branch', 'target_branch', 'title'), optional=('allow_collaboration', 'allow_maintainer_to_push', 'approvals_before_merge', 'assignee_id', 'assignee_ids', 'description', 'labels', 'milestone_id', 'remove_source_branch', 'reviewer_ids', 'squash', 'target_project_id'))
    _update_attrs = RequiredOptional(optional=('target_branch', 'assignee_id', 'title', 'description', 'state_event', 'labels', 'milestone_id', 'remove_source_branch', 'discussion_locked', 'allow_maintainer_to_push', 'squash', 'reviewer_ids'))
    _list_filters = ('state', 'order_by', 'sort', 'milestone', 'view', 'labels', 'created_after', 'created_before', 'updated_after', 'updated_before', 'scope', 'iids', 'author_id', 'assignee_id', 'approver_ids', 'approved_by_ids', 'my_reaction_emoji', 'source_branch', 'target_branch', 'search', 'wip')
    _types = {'approver_ids': types.ArrayAttribute, 'approved_by_ids': types.ArrayAttribute, 'iids': types.ArrayAttribute, 'labels': types.CommaSeparatedListAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectMergeRequest:
        return cast(ProjectMergeRequest, super().get(id=id, lazy=lazy, **kwargs))