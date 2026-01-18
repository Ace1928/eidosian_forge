import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class HeadMixin(_RestManagerBase):

    @exc.on_http_error(exc.GitlabHeadError)
    def head(self, id: Optional[Union[str, int]]=None, **kwargs: Any) -> 'requests.structures.CaseInsensitiveDict[Any]':
        """Retrieve headers from an endpoint.

        Args:
            id: ID of the object to retrieve
            **kwargs: Extra options to send to the server (e.g. sudo)

        Returns:
            A requests header object.

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabHeadError: If the server cannot perform the request
        """
        if TYPE_CHECKING:
            assert self.path is not None
        path = self.path
        if id is not None:
            path = f'{path}/{utils.EncodedId(id)}'
        return self.gitlab.http_head(path, **kwargs)