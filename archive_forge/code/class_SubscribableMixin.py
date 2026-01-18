import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class SubscribableMixin(_RestObjectBase):
    _id_attr: Optional[str]
    _attrs: Dict[str, Any]
    _module: ModuleType
    _parent_attrs: Dict[str, Any]
    _updated_attrs: Dict[str, Any]
    manager: base.RESTManager

    @cli.register_custom_action(('ProjectIssue', 'ProjectMergeRequest', 'ProjectLabel', 'GroupLabel'))
    @exc.on_http_error(exc.GitlabSubscribeError)
    def subscribe(self, **kwargs: Any) -> None:
        """Subscribe to the object notifications.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabSubscribeError: If the subscription cannot be done
        """
        path = f'{self.manager.path}/{self.encoded_id}/subscribe'
        server_data = self.manager.gitlab.http_post(path, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(server_data, requests.Response)
        self._update_attrs(server_data)

    @cli.register_custom_action(('ProjectIssue', 'ProjectMergeRequest', 'ProjectLabel', 'GroupLabel'))
    @exc.on_http_error(exc.GitlabUnsubscribeError)
    def unsubscribe(self, **kwargs: Any) -> None:
        """Unsubscribe from the object notifications.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUnsubscribeError: If the unsubscription cannot be done
        """
        path = f'{self.manager.path}/{self.encoded_id}/unsubscribe'
        server_data = self.manager.gitlab.http_post(path, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(server_data, requests.Response)
        self._update_attrs(server_data)