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
class ProjectIssue(UserAgentDetailMixin, SubscribableMixin, TodoMixin, TimeTrackingMixin, ParticipantsMixin, SaveMixin, ObjectDeleteMixin, RESTObject):
    _repr_attr = 'title'
    _id_attr = 'iid'
    awardemojis: ProjectIssueAwardEmojiManager
    discussions: ProjectIssueDiscussionManager
    links: 'ProjectIssueLinkManager'
    notes: ProjectIssueNoteManager
    resourcelabelevents: ProjectIssueResourceLabelEventManager
    resourcemilestoneevents: ProjectIssueResourceMilestoneEventManager
    resourcestateevents: ProjectIssueResourceStateEventManager
    resource_iteration_events: ProjectIssueResourceIterationEventManager
    resource_weight_events: ProjectIssueResourceWeightEventManager

    @cli.register_custom_action('ProjectIssue', ('to_project_id',))
    @exc.on_http_error(exc.GitlabUpdateError)
    def move(self, to_project_id: int, **kwargs: Any) -> None:
        """Move the issue to another project.

        Args:
            to_project_id: ID of the target project
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUpdateError: If the issue could not be moved
        """
        path = f'{self.manager.path}/{self.encoded_id}/move'
        data = {'to_project_id': to_project_id}
        server_data = self.manager.gitlab.http_post(path, post_data=data, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(server_data, dict)
        self._update_attrs(server_data)

    @cli.register_custom_action('ProjectIssue', ('move_after_id', 'move_before_id'))
    @exc.on_http_error(exc.GitlabUpdateError)
    def reorder(self, move_after_id: Optional[int]=None, move_before_id: Optional[int]=None, **kwargs: Any) -> None:
        """Reorder an issue on a board.

        Args:
            move_after_id: ID of an issue that should be placed after this issue
            move_before_id: ID of an issue that should be placed before this issue
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUpdateError: If the issue could not be reordered
        """
        path = f'{self.manager.path}/{self.encoded_id}/reorder'
        data: Dict[str, Any] = {}
        if move_after_id is not None:
            data['move_after_id'] = move_after_id
        if move_before_id is not None:
            data['move_before_id'] = move_before_id
        server_data = self.manager.gitlab.http_put(path, post_data=data, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(server_data, dict)
        self._update_attrs(server_data)

    @cli.register_custom_action('ProjectIssue')
    @exc.on_http_error(exc.GitlabGetError)
    def related_merge_requests(self, **kwargs: Any) -> Dict[str, Any]:
        """List merge requests related to the issue.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetErrot: If the merge requests could not be retrieved

        Returns:
            The list of merge requests.
        """
        path = f'{self.manager.path}/{self.encoded_id}/related_merge_requests'
        result = self.manager.gitlab.http_get(path, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(result, dict)
        return result

    @cli.register_custom_action('ProjectIssue')
    @exc.on_http_error(exc.GitlabGetError)
    def closed_by(self, **kwargs: Any) -> Dict[str, Any]:
        """List merge requests that will close the issue when merged.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetErrot: If the merge requests could not be retrieved

        Returns:
            The list of merge requests.
        """
        path = f'{self.manager.path}/{self.encoded_id}/closed_by'
        result = self.manager.gitlab.http_get(path, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(result, dict)
        return result