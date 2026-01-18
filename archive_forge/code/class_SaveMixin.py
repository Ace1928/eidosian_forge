import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class SaveMixin(_RestObjectBase):
    """Mixin for RESTObject's that can be updated."""
    _id_attr: Optional[str]
    _attrs: Dict[str, Any]
    _module: ModuleType
    _parent_attrs: Dict[str, Any]
    _updated_attrs: Dict[str, Any]
    manager: base.RESTManager

    def _get_updated_data(self) -> Dict[str, Any]:
        updated_data = {}
        for attr in self.manager._update_attrs.required:
            updated_data[attr] = getattr(self, attr)
        updated_data.update(self._updated_attrs)
        return updated_data

    def save(self, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Save the changes made to the object to the server.

        The object is updated to match what the server returns.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Returns:
            The new object data (*not* a RESTObject)

        Raise:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUpdateError: If the server cannot perform the request
        """
        updated_data = self._get_updated_data()
        if not updated_data:
            return None
        obj_id = self.encoded_id
        if TYPE_CHECKING:
            assert isinstance(self.manager, UpdateMixin)
        server_data = self.manager.update(obj_id, updated_data, **kwargs)
        self._update_attrs(server_data)
        return server_data