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
class ProjectIssueLinkManager(ListMixin, CreateMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/issues/{issue_iid}/links'
    _obj_cls = ProjectIssueLink
    _from_parent_attrs = {'project_id': 'project_id', 'issue_iid': 'iid'}
    _create_attrs = RequiredOptional(required=('target_project_id', 'target_issue_iid'))

    @exc.on_http_error(exc.GitlabCreateError)
    def create(self, data: Dict[str, Any], **kwargs: Any) -> Tuple[RESTObject, RESTObject]:
        """Create a new object.

        Args:
            data: parameters to send to the server to create the
                         resource
            **kwargs: Extra options to send to the server (e.g. sudo)

        Returns:
            The source and target issues

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the server cannot perform the request
        """
        self._create_attrs.validate_attrs(data=data)
        if TYPE_CHECKING:
            assert self.path is not None
        server_data = self.gitlab.http_post(self.path, post_data=data, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(server_data, dict)
            assert self._parent is not None
        source_issue = ProjectIssue(self._parent.manager, server_data['source_issue'])
        target_issue = ProjectIssue(self._parent.manager, server_data['target_issue'])
        return (source_issue, target_issue)