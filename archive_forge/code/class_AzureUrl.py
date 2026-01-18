from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os
import re
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
class AzureUrl(CloudUrl):
    """CloudUrl subclass for Azure's unique blob storage URL structure.

    Attributes:
      scheme (ProviderPrefix): AZURE (http) or AZURE_TLS (https).
      bucket_name (str|None): Storage container name in URL.
      object_name (str|None): Storage object name in URL.
      generation (str|None): Equivalent to Azure 'versionId'. Datetime string.
      snapshot (str|None): Similar to 'versionId'. URL parameter used to capture
        a specific version of a storage object. Datetime string.
      account (str): Account owning storage resource.
  """

    def __init__(self, scheme, bucket_name=None, object_name=None, generation=None, snapshot=None, account=None):
        super(AzureUrl, self).__init__(scheme, bucket_name, object_name, generation)
        self.snapshot = snapshot if snapshot else None
        if not account:
            raise errors.InvalidUrlError('Azure URLs must contain an account name.')
        self.account = account

    @classmethod
    def from_url_string(cls, url_string):
        """Parses the url string and return the storage URL object.

    Args:
      url_string (str): Azure storage URL of the form:
        http://account.blob.core.windows.net/container/blob

    Returns:
      AzureUrl object

    Raises:
      InvalidUrlError: Raised if the url_string is not a valid cloud URL.
    """
        scheme = _get_scheme_from_url_string(url_string)
        AzureUrl.validate_url_string(url_string, scheme)
        schemeless_url_string = url_string[len(scheme.value + SCHEME_DELIMITER):]
        hostname, _, path_and_params = schemeless_url_string.partition(CLOUD_URL_DELIMITER)
        account, _, _ = hostname.partition('.')
        container, _, blob_and_params = path_and_params.partition(CLOUD_URL_DELIMITER)
        blob, _, params = blob_and_params.partition('?')
        params_dict = urllib.parse.parse_qs(params)
        return cls(scheme, bucket_name=container, object_name=blob, generation=params_dict['versionId'][0] if 'versionId' in params_dict else None, snapshot=params_dict['snapshot'][0] if 'snapshot' in params_dict else None, account=account)

    @classmethod
    def is_valid_scheme(cls, scheme):
        return scheme in VALID_HTTP_SCHEMES

    def _validate_scheme(self):
        if not AzureUrl.is_valid_scheme(self.scheme):
            raise errors.InvalidUrlError('Invalid Azure scheme "{}"'.format(self.scheme))

    @classmethod
    def validate_url_string(cls, url_string, scheme):
        AzureUrl.is_valid_scheme(scheme)
        if not (AZURE_DOMAIN in url_string and AzureUrl.is_valid_scheme(scheme)):
            raise errors.InvalidUrlError('Invalid Azure URL: "{}"'.format(url_string))

    @property
    def url_string(self):
        url_parts = list(urllib.parse.urlsplit(self.versionless_url_string))
        url_parameters = {}
        if self.generation:
            url_parameters['versionId'] = self.generation
        if self.snapshot:
            url_parameters['snapshot'] = self.snapshot
        url_parts[3] = urllib.parse.urlencode(url_parameters)
        return urllib.parse.urlunsplit(url_parts)

    @property
    def versionless_url_string(self):
        if self.is_provider():
            return '{}{}{}.{}'.format(self.scheme.value, SCHEME_DELIMITER, self.account, AZURE_DOMAIN)
        elif self.is_bucket():
            return '{}{}{}.{}/{}'.format(self.scheme.value, SCHEME_DELIMITER, self.account, AZURE_DOMAIN, self.bucket_name)
        return '{}{}{}.{}/{}/{}'.format(self.scheme.value, SCHEME_DELIMITER, self.account, AZURE_DOMAIN, self.bucket_name, self.object_name)