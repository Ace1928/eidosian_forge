from typing import Any, cast, Dict, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ListMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
Enable a deploy key for a project.

        Args:
            key_id: The ID of the key to enable
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabProjectDeployKeyError: If the key could not be enabled

        Returns:
            A dict of the result.
        