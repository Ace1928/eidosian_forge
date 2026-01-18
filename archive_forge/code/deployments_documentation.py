from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, RetrieveMixin, SaveMixin, UpdateMixin
from gitlab.types import RequiredOptional
from .merge_requests import ProjectDeploymentMergeRequestManager  # noqa: F401
Approve or reject a blocked deployment.

        Args:
            status: Either "approved" or "rejected"
            comment: A comment to go with the approval
            represented_as: The name of the User/Group/Role to use for the
                            approval, when the user belongs to multiple
                            approval rules.
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabMRApprovalError: If the approval failed

        Returns:
           A dict containing the result.

        https://docs.gitlab.com/ee/api/deployments.html#approve-or-reject-a-blocked-deployment
        