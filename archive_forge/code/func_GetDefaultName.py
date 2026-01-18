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