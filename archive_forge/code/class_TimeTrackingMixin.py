import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class TimeTrackingMixin(_RestObjectBase):
    _id_attr: Optional[str]
    _attrs: Dict[str, Any]
    _module: ModuleType
    _parent_attrs: Dict[str, Any]
    _updated_attrs: Dict[str, Any]
    manager: base.RESTManager

    @cli.register_custom_action(('ProjectIssue', 'ProjectMergeRequest'))
    @exc.on_http_error(exc.GitlabTimeTrackingError)
    def time_stats(self, **kwargs: Any) -> Dict[str, Any]:
        """Get time stats for the object.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabTimeTrackingError: If the time tracking update cannot be done
        """
        if 'time_stats' in self.attributes:
            time_stats = self.attributes['time_stats']
            if TYPE_CHECKING:
                assert isinstance(time_stats, dict)
            return time_stats
        path = f'{self.manager.path}/{self.encoded_id}/time_stats'
        result = self.manager.gitlab.http_get(path, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(result, requests.Response)
        return result

    @cli.register_custom_action(('ProjectIssue', 'ProjectMergeRequest'), ('duration',))
    @exc.on_http_error(exc.GitlabTimeTrackingError)
    def time_estimate(self, duration: str, **kwargs: Any) -> Dict[str, Any]:
        """Set an estimated time of work for the object.

        Args:
            duration: Duration in human format (e.g. 3h30)
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabTimeTrackingError: If the time tracking update cannot be done
        """
        path = f'{self.manager.path}/{self.encoded_id}/time_estimate'
        data = {'duration': duration}
        result = self.manager.gitlab.http_post(path, post_data=data, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(result, requests.Response)
        return result

    @cli.register_custom_action(('ProjectIssue', 'ProjectMergeRequest'))
    @exc.on_http_error(exc.GitlabTimeTrackingError)
    def reset_time_estimate(self, **kwargs: Any) -> Dict[str, Any]:
        """Resets estimated time for the object to 0 seconds.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabTimeTrackingError: If the time tracking update cannot be done
        """
        path = f'{self.manager.path}/{self.encoded_id}/reset_time_estimate'
        result = self.manager.gitlab.http_post(path, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(result, requests.Response)
        return result

    @cli.register_custom_action(('ProjectIssue', 'ProjectMergeRequest'), ('duration',))
    @exc.on_http_error(exc.GitlabTimeTrackingError)
    def add_spent_time(self, duration: str, **kwargs: Any) -> Dict[str, Any]:
        """Add time spent working on the object.

        Args:
            duration: Duration in human format (e.g. 3h30)
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabTimeTrackingError: If the time tracking update cannot be done
        """
        path = f'{self.manager.path}/{self.encoded_id}/add_spent_time'
        data = {'duration': duration}
        result = self.manager.gitlab.http_post(path, post_data=data, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(result, requests.Response)
        return result

    @cli.register_custom_action(('ProjectIssue', 'ProjectMergeRequest'))
    @exc.on_http_error(exc.GitlabTimeTrackingError)
    def reset_spent_time(self, **kwargs: Any) -> Dict[str, Any]:
        """Resets the time spent working on the object.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabTimeTrackingError: If the time tracking update cannot be done
        """
        path = f'{self.manager.path}/{self.encoded_id}/reset_spent_time'
        result = self.manager.gitlab.http_post(path, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(result, requests.Response)
        return result