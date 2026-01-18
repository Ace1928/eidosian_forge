import logging
import os
from types import SimpleNamespace
from rasterio._path import _parse_path, _UnparsedPath
class AzureSession(Session):
    """Configures access to secured resources stored in Microsoft Azure Blob Storage.
    """

    def __init__(self, azure_storage_connection_string=None, azure_storage_account=None, azure_storage_access_key=None, azure_unsigned=False):
        """Create new Microsoft Azure Blob Storage session

        Parameters
        ----------
        azure_storage_connection_string: string
            A connection string contains both an account name and a secret key.
        azure_storage_account: string
            An account name
        azure_storage_access_key: string
            A secret key
        azure_unsigned : bool, optional (default: False)
            If True, requests will be unsigned.
        """
        self.unsigned = parse_bool(os.getenv('AZURE_NO_SIGN_REQUEST', azure_unsigned))
        self.storage_account = azure_storage_account or os.getenv('AZURE_STORAGE_ACCOUNT')
        self.storage_access_key = azure_storage_access_key or os.getenv('AZURE_STORAGE_ACCESS_KEY')
        if azure_storage_connection_string:
            self._creds = {'azure_storage_connection_string': azure_storage_connection_string}
        elif not self.unsigned:
            self._creds = {'azure_storage_account': self.storage_account, 'azure_storage_access_key': self.storage_access_key}
        else:
            self._creds = {'azure_storage_account': self.storage_account}

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
        return 'AZURE_STORAGE_CONNECTION_STRING' in config or ('AZURE_STORAGE_ACCOUNT' in config and 'AZURE_STORAGE_ACCESS_KEY' in config) or ('AZURE_STORAGE_ACCOUNT' in config and 'AZURE_NO_SIGN_REQUEST' in config)

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
        if self.unsigned:
            return {'AZURE_NO_SIGN_REQUEST': 'YES', 'AZURE_STORAGE_ACCOUNT': self.storage_account}
        else:
            return {k.upper(): v for k, v in self.credentials.items()}