from typing import Any, cast, Dict, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
class ProjectEnvironment(SaveMixin, ObjectDeleteMixin, RESTObject):

    @cli.register_custom_action('ProjectEnvironment')
    @exc.on_http_error(exc.GitlabStopError)
    def stop(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
        """Stop the environment.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabStopError: If the operation failed

        Returns:
           A dict of the result.
        """
        path = f'{self.manager.path}/{self.encoded_id}/stop'
        return self.manager.gitlab.http_post(path, **kwargs)