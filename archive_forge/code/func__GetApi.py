from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import boto
from boto import config
from gslib import context_config
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import CloudApi
from gslib.cs_api_map import ApiMapConstants
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.utils import boto_util
def _GetApi(self, provider):
    """Returns a valid CloudApi for use by the caller.

    This function lazy-loads connection and credentials using the API map
    and credential store provided during class initialization.

    Args:
      provider: Provider to load API for. If None, class-wide default is used.

    Raises:
      ArgumentException if there is no matching API available in the API map.

    Returns:
      Valid API instance that can be used to communicate with the Cloud
      Storage provider.
    """
    provider = provider or self.provider
    if not provider:
        raise ArgumentException('No provider selected for _GetApi')
    provider = str(provider)
    if provider not in self.loaded_apis:
        self.loaded_apis[provider] = {}
    api_selector = self.GetApiSelector(provider)
    if api_selector not in self.loaded_apis[provider]:
        self._LoadApi(provider, api_selector)
    return self.loaded_apis[provider][api_selector]