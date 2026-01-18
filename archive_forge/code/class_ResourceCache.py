from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import file_cache
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
class ResourceCache(PERSISTENT_CACHE_IMPLEMENTATION.Cache):
    """A resource cache object."""

    def __init__(self, name=None, create=True):
        """ResourceCache constructor.

    Args:
      name: The persistent cache object name. If None then a default name
        conditioned on the account name is used.
          <GLOBAL_CONFIG_DIR>/cache/<ACCOUNT>/resource.cache
      create: Create the cache if it doesn't exist if True.
    """
        if not name:
            name = self.GetDefaultName()
        super(ResourceCache, self).__init__(name=name, create=create, version=VERSION)

    @staticmethod
    def GetDefaultName():
        """Returns the default resource cache name."""
        path = [config.Paths().cache_dir]
        account = properties.VALUES.core.account.Get(required=False)
        if account:
            path.append(account)
        files.MakeDir(os.path.join(*path))
        path.append('resource.cache')
        return os.path.join(*path)