import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class UploadMixin(_RestObjectBase):
    _id_attr: Optional[str]
    _attrs: Dict[str, Any]
    _module: ModuleType
    _parent_attrs: Dict[str, Any]
    _updated_attrs: Dict[str, Any]
    _upload_path: str
    manager: base.RESTManager

    def _get_upload_path(self) -> str:
        """Formats _upload_path with object attributes.

        Returns:
            The upload path
        """
        if TYPE_CHECKING:
            assert isinstance(self._upload_path, str)
        data = self.attributes
        return self._upload_path.format(**data)

    @cli.register_custom_action(('Project', 'ProjectWiki'), ('filename', 'filepath'))
    @exc.on_http_error(exc.GitlabUploadError)
    def upload(self, filename: str, filedata: Optional[bytes]=None, filepath: Optional[str]=None, **kwargs: Any) -> Dict[str, Any]:
        """Upload the specified file.

        .. note::

            Either ``filedata`` or ``filepath`` *MUST* be specified.

        Args:
            filename: The name of the file being uploaded
            filedata: The raw data of the file being uploaded
            filepath: The path to a local file to upload (optional)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUploadError: If the file upload fails
            GitlabUploadError: If ``filedata`` and ``filepath`` are not
                specified
            GitlabUploadError: If both ``filedata`` and ``filepath`` are
                specified

        Returns:
            A ``dict`` with info on the uploaded file
        """
        if filepath is None and filedata is None:
            raise exc.GitlabUploadError('No file contents or path specified')
        if filedata is not None and filepath is not None:
            raise exc.GitlabUploadError('File contents and file path specified')
        if filepath is not None:
            with open(filepath, 'rb') as f:
                filedata = f.read()
        file_info = {'file': (filename, filedata)}
        path = self._get_upload_path()
        server_data = self.manager.gitlab.http_post(path, files=file_info, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(server_data, dict)
        return server_data