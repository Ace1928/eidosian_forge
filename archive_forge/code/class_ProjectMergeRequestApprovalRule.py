from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectMergeRequestApprovalRule(SaveMixin, ObjectDeleteMixin, RESTObject):
    _repr_attr = 'name'
    id: int
    approval_rule_id: int
    merge_request_iid: int

    @exc.on_http_error(exc.GitlabUpdateError)
    def save(self, **kwargs: Any) -> None:
        """Save the changes made to the object to the server.

        The object is updated to match what the server returns.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raise:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUpdateError: If the server cannot perform the request
        """
        self.approval_rule_id = self.id
        self.merge_request_iid = self._parent_attrs['mr_iid']
        self.id = self._parent_attrs['project_id']
        SaveMixin.save(self, **kwargs)