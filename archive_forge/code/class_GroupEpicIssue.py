from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .events import GroupEpicResourceLabelEventManager  # noqa: F401
from .notes import GroupEpicNoteManager  # noqa: F401
class GroupEpicIssue(ObjectDeleteMixin, SaveMixin, RESTObject):
    _id_attr = 'epic_issue_id'
    manager: 'GroupEpicIssueManager'

    def save(self, **kwargs: Any) -> None:
        """Save the changes made to the object to the server.

        The object is updated to match what the server returns.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raise:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUpdateError: If the server cannot perform the request
        """
        updated_data = self._get_updated_data()
        if not updated_data:
            return
        obj_id = self.encoded_id
        self.manager.update(obj_id, updated_data, **kwargs)