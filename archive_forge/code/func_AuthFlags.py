from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.web_security_scanner import wss_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
def AuthFlags():
    """Hook to add additional authentication flags.

  Returns:
    Auth flag group
  """
    auth_group = base.ArgumentGroup(mutex=False)
    auth_group.AddArgument(_TypeFlag())
    auth_group.AddArgument(_UserFlag())
    auth_group.AddArgument(_PasswordFlag())
    auth_group.AddArgument(_UrlFlag())
    return [auth_group]