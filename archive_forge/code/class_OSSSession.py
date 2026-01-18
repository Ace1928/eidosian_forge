import logging
import os
from types import SimpleNamespace
from rasterio._path import _parse_path, _UnparsedPath
class OSSSession(Session):
    """Configures access to secured resources stored in Alibaba Cloud OSS.
    """

    def __init__(self, oss_access_key_id=None, oss_secret_access_key=None, oss_endpoint=None):
        """Create new Alibaba Cloud OSS session

        Parameters
        ----------
        oss_access_key_id: string, optional (default: None)
            An access key id
        oss_secret_access_key: string, optional (default: None)
            An secret access key
        oss_endpoint: string, optional (default: None)
            the region attached to the bucket
        """
        self._creds = {'oss_access_key_id': oss_access_key_id, 'oss_secret_access_key': oss_secret_access_key, 'oss_endpoint': oss_endpoint}

    @classmethod
    def hascreds(cls, config):
        """Determine if the given configuration has proper credentials

        Parameters
        ----------
        cls : class
            A Session class.
        config : dict
            GDAL configuration as a dict.

        Returns
        -------
        bool

        """
        return 'OSS_ACCESS_KEY_ID' in config and 'OSS_SECRET_ACCESS_KEY' in config

    @property
    def credentials(self):
        """The session credentials as a dict"""
        return self._creds

    def get_credential_options(self):
        """Get credentials as GDAL configuration options

        Returns
        -------
        dict

        """
        return {k.upper(): v for k, v in self.credentials.items()}