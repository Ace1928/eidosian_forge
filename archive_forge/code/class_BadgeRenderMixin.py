import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class BadgeRenderMixin(_RestManagerBase):

    @cli.register_custom_action(('GroupBadgeManager', 'ProjectBadgeManager'), ('link_url', 'image_url'))
    @exc.on_http_error(exc.GitlabRenderError)
    def render(self, link_url: str, image_url: str, **kwargs: Any) -> Dict[str, Any]:
        """Preview link_url and image_url after interpolation.

        Args:
            link_url: URL of the badge link
            image_url: URL of the badge image
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabRenderError: If the rendering failed

        Returns:
            The rendering properties
        """
        path = f'{self.path}/render'
        data = {'link_url': link_url, 'image_url': image_url}
        result = self.gitlab.http_get(path, data, **kwargs)
        if TYPE_CHECKING:
            assert not isinstance(result, requests.Response)
        return result