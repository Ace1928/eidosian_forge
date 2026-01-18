import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class RefreshMixin(_RestObjectBase):
    _id_attr: Optional[str]
    _attrs: Dict[str, Any]
    _module: ModuleType
    _parent_attrs: Dict[str, Any]
    _updated_attrs: Dict[str, Any]
    manager: base.RESTManager

    @exc.on_http_error(exc.GitlabGetError)
    def refresh(self, **kwargs: Any) -> None:
        """Refresh a single object from server.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Returns None (updates the object)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server cannot perform the request
        """
        if self._id_attr:
            path = f'{self.manager.path}/{self.encoded_id}'
        else:
            if TYPE_CHECKING:
                assert self.manager.path is not None
            path = self.manager.path
        server_data = self.manager.gitlab.http_get(path, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(server_data, requests.Response)
        self._update_attrs(server_data)