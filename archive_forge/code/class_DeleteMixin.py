import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class DeleteMixin(_RestManagerBase):
    _computed_path: Optional[str]
    _from_parent_attrs: Dict[str, Any]
    _obj_cls: Optional[Type[base.RESTObject]]
    _parent: Optional[base.RESTObject]
    _parent_attrs: Dict[str, Any]
    _path: Optional[str]
    gitlab: gitlab.Gitlab

    @exc.on_http_error(exc.GitlabDeleteError)
    def delete(self, id: Optional[Union[str, int]]=None, **kwargs: Any) -> None:
        """Delete an object on the server.

        Args:
            id: ID of the object to delete
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabDeleteError: If the server cannot perform the request
        """
        if id is None:
            path = self.path
        else:
            path = f'{self.path}/{utils.EncodedId(id)}'
        if TYPE_CHECKING:
            assert path is not None
        self.gitlab.http_delete(path, **kwargs)