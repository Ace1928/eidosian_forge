import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class RotateMixin(_RestManagerBase):
    _computed_path: Optional[str]
    _from_parent_attrs: Dict[str, Any]
    _obj_cls: Optional[Type[base.RESTObject]]
    _parent: Optional[base.RESTObject]
    _parent_attrs: Dict[str, Any]
    _path: Optional[str]
    gitlab: gitlab.Gitlab

    @exc.on_http_error(exc.GitlabRotateError)
    def rotate(self, id: Union[str, int], expires_at: Optional[str]=None, **kwargs: Any) -> Dict[str, Any]:
        """Rotate an access token.

        Args:
            id: ID of the token to rotate
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabRotateError: If the server cannot perform the request
        """
        path = f'{self.path}/{utils.EncodedId(id)}/rotate'
        data: Dict[str, Any] = {}
        if expires_at is not None:
            data = {'expires_at': expires_at}
        server_data = self.gitlab.http_post(path, post_data=data, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(server_data, requests.Response)
        return server_data