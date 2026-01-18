import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class GetWithoutIdMixin(HeadMixin, _RestManagerBase):
    _computed_path: Optional[str]
    _from_parent_attrs: Dict[str, Any]
    _obj_cls: Optional[Type[base.RESTObject]]
    _optional_get_attrs: Tuple[str, ...] = ()
    _parent: Optional[base.RESTObject]
    _parent_attrs: Dict[str, Any]
    _path: Optional[str]
    gitlab: gitlab.Gitlab

    @exc.on_http_error(exc.GitlabGetError)
    def get(self, **kwargs: Any) -> base.RESTObject:
        """Retrieve a single object.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Returns:
            The generated RESTObject

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server cannot perform the request
        """
        if TYPE_CHECKING:
            assert self.path is not None
        server_data = self.gitlab.http_get(self.path, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(server_data, requests.Response)
            assert self._obj_cls is not None
        return self._obj_cls(self, server_data)