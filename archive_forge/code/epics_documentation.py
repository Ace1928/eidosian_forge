from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .events import GroupEpicResourceLabelEventManager  # noqa: F401
from .notes import GroupEpicNoteManager  # noqa: F401
Create a new object.

        Args:
            data: Parameters to send to the server to create the
                         resource
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the server cannot perform the request

        Returns:
            A new instance of the manage object class build with
                the data sent by the server
        