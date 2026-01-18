from typing import Any, cast, Dict, Optional, Union
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class GroupLabel(SubscribableMixin, SaveMixin, ObjectDeleteMixin, RESTObject):
    _id_attr = 'name'
    manager: 'GroupLabelManager'

    @exc.on_http_error(exc.GitlabUpdateError)
    def save(self, **kwargs: Any) -> None:
        """Saves the changes made to the object to the server.

        The object is updated to match what the server returns.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct.
            GitlabUpdateError: If the server cannot perform the request.
        """
        updated_data = self._get_updated_data()
        server_data = self.manager.update(None, updated_data, **kwargs)
        self._update_attrs(server_data)