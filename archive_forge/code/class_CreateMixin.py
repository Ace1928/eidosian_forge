import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class CreateMixin(_RestManagerBase):
    _computed_path: Optional[str]
    _from_parent_attrs: Dict[str, Any]
    _obj_cls: Optional[Type[base.RESTObject]]
    _parent: Optional[base.RESTObject]
    _parent_attrs: Dict[str, Any]
    _path: Optional[str]
    gitlab: gitlab.Gitlab

    @exc.on_http_error(exc.GitlabCreateError)
    def create(self, data: Optional[Dict[str, Any]]=None, **kwargs: Any) -> base.RESTObject:
        """Create a new object.

        Args:
            data: parameters to send to the server to create the
                         resource
            **kwargs: Extra options to send to the server (e.g. sudo)

        Returns:
            A new instance of the managed object class built with
                the data sent by the server

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the server cannot perform the request
        """
        if data is None:
            data = {}
        self._create_attrs.validate_attrs(data=data)
        data, files = utils._transform_types(data=data, custom_types=self._types, transform_data=False)
        path = kwargs.pop('path', self.path)
        server_data = self.gitlab.http_post(path, post_data=data, files=files, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(server_data, requests.Response)
            assert self._obj_cls is not None
        return self._obj_cls(self, server_data)