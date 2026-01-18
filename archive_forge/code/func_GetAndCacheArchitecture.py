from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import platform
import re
import time
import uuid
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
def GetAndCacheArchitecture(user_platform):
    """Get and cache architecture of client machine.

  For M1 Macs running x86_64 Python using Rosetta, user_platform.architecture
  (from platform.machine()) returns x86_64. We can use
  IsActuallyM1ArmArchitecture() to determine the underlying hardware; however,
  it requires a system call that might take ~5ms.
  To mitigate this, we will persist this value as an internal property with
  INSTALLATION scope.

  Args:
    user_platform: platforms.Platform.Current()

  Returns:
    client machine architecture
  """
    active_config_store = config.GetConfigStore()
    if active_config_store and active_config_store.Get('client_arch'):
        return active_config_store.Get('client_arch')
    if user_platform.operating_system == platforms.OperatingSystem.MACOSX and user_platform.architecture == platforms.Architecture.x86_64 and platforms.Platform.IsActuallyM1ArmArchitecture():
        arch = '{}_{}'.format(platforms.Architecture.x86_64, platforms.Architecture.arm)
    else:
        arch = str(user_platform.architecture)
    if active_config_store:
        active_config_store.Set('client_arch', arch)
    return arch