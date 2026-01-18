import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class GetMixin(HeadMixin, _RestManagerBase):
    _computed_path: Optional[str]
    _from_parent_attrs: Dict[str, Any]
    _obj_cls: Optional[Type[base.RESTObject]]
    _optional_get_attrs: Tuple[str, ...] = ()
    _parent: Optional[base.RESTObject]
    _parent_attrs: Dict[str, Any]
    _path: Optional[str]
    gitlab: gitlab.Gitlab

    @exc.on_http_error(exc.GitlabGetError)
    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> base.RESTObject:
        """Retrieve a single object.

        Args:
            id: ID of the object to retrieve
            lazy: If True, don't request the server, but create a
                         shallow object giving access to the managers. This is
                         useful if you want to avoid useless calls to the API.
            **kwargs: Extra options to send to the server (e.g. sudo)

        Returns:
            The generated RESTObject.

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server cannot perform the request
        """
        if isinstance(id, str):
            id = utils.EncodedId(id)
        path = f'{self.path}/{id}'
        if TYPE_CHECKING:
            assert self._obj_cls is not None
        if lazy is True:
            if TYPE_CHECKING:
                assert self._obj_cls._id_attr is not None
            return self._obj_cls(self, {self._obj_cls._id_attr: id}, lazy=lazy)
        server_data = self.gitlab.http_get(path, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(server_data, requests.Response)
        return self._obj_cls(self, server_data, lazy=lazy)