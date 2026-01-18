import base64
import urllib
import requests
import requests.exceptions
from requests.adapters import HTTPAdapter, Retry
from fsspec import AbstractFileSystem
from fsspec.spec import AbstractBufferedFile
def _close_handle(self, handle):
    """
        Close a handle, which was opened by :func:`_create_handle`.

        Parameters
        ----------
        handle: str
            Which handle to close.
        """
    try:
        self._send_to_api(method='post', endpoint='close', json={'handle': handle})
    except DatabricksException as e:
        if e.error_code == 'RESOURCE_DOES_NOT_EXIST':
            raise FileNotFoundError(e.message)
        raise e