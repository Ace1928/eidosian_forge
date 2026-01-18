from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.organizations import org_utils
import six
def ConvertOrgArgToObfuscatedCustomerId(org_arg):
    """Convert organization argument to obfuscated customer id.

  Args:
    org_arg: organization argument

  Returns:
    Obfuscated customer id

  Example:
    org_id: 12345
    organization_obj:
    {
      owner: {
        directoryCustomerId: A08w1n5gg
      }
    }
  """
    organization_obj = org_utils.GetOrganization(org_arg)
    if organization_obj:
        return organization_obj.owner.directoryCustomerId
    else:
        raise org_utils.UnknownOrganizationError(org_arg, metavar='ORGANIZATION')