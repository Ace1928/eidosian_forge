from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.media.asset import utils
from googlecloudsdk.core import resources
def GetExistingResource(api_version, request_message):
    """Get the modified resource.

  Args:
    api_version: the request's release track.
    request_message: request message type in the python client.

  Returns:
    The modified resource.
  """
    return utils.GetClient(api_version).projects_locations_assetTypes_assets.Get(request_message)