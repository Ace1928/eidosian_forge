import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class UpdateMixin(_RestManagerBase):
    _computed_path: Optional[str]
    _from_parent_attrs: Dict[str, Any]
    _obj_cls: Optional[Type[base.RESTObject]]
    _parent: Optional[base.RESTObject]
    _parent_attrs: Dict[str, Any]
    _path: Optional[str]
    _update_method: UpdateMethod = UpdateMethod.PUT
    gitlab: gitlab.Gitlab

    def _get_update_method(self) -> Callable[..., Union[Dict[str, Any], requests.Response]]:
        """Return the HTTP method to use.

        Returns:
            http_put (default) or http_post
        """
        if self._update_method is UpdateMethod.POST:
            http_method = self.gitlab.http_post
        elif self._update_method is UpdateMethod.PATCH:
            http_method = self.gitlab.http_patch
        else:
            http_method = self.gitlab.http_put
        return http_method

    @exc.on_http_error(exc.GitlabUpdateError)
    def update(self, id: Optional[Union[str, int]]=None, new_data: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Dict[str, Any]:
        """Update an object on the server.

        Args:
            id: ID of the object to update (can be None if not required)
            new_data: the update data for the object
            **kwargs: Extra options to send to the server (e.g. sudo)

        Returns:
            The new object data (*not* a RESTObject)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUpdateError: If the server cannot perform the request
        """
        new_data = new_data or {}
        if id is None:
            path = self.path
        else:
            path = f'{self.path}/{utils.EncodedId(id)}'
        excludes = []
        if self._obj_cls is not None and self._obj_cls._id_attr is not None:
            excludes = [self._obj_cls._id_attr]
        self._update_attrs.validate_attrs(data=new_data, excludes=excludes)
        new_data, files = utils._transform_types(data=new_data, custom_types=self._types, transform_data=False)
        http_method = self._get_update_method()
        result = http_method(path, post_data=new_data, files=files, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(result, requests.Response)
        return result