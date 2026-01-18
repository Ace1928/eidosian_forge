import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class DownloadMixin(_RestObjectBase):
    _id_attr: Optional[str]
    _attrs: Dict[str, Any]
    _module: ModuleType
    _parent_attrs: Dict[str, Any]
    _updated_attrs: Dict[str, Any]
    manager: base.RESTManager

    @cli.register_custom_action(('GroupExport', 'ProjectExport'))
    @exc.on_http_error(exc.GitlabGetError)
    def download(self, streamed: bool=False, action: Optional[Callable[[bytes], None]]=None, chunk_size: int=1024, *, iterator: bool=False, **kwargs: Any) -> Optional[Union[bytes, Iterator[Any]]]:
        """Download the archive of a resource export.

        Args:
            streamed: If True the data will be processed by chunks of
                `chunk_size` and each chunk is passed to `action` for
                treatment
            iterator: If True directly return the underlying response
                iterator
            action: Callable responsible of dealing with chunk of
                data
            chunk_size: Size of each chunk
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            The blob content if streamed is False, None otherwise
        """
        path = f'{self.manager.path}/download'
        result = self.manager.gitlab.http_get(path, streamed=streamed, raw=True, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(result, requests.Response)
        return utils.response_content(result, streamed, action, chunk_size, iterator=iterator)