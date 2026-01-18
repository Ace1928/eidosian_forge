from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_api
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.network_connectivity import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
Deactivate a spoke.

  Deactivate the specified spoke. When you deactivate a spoke, it
  can't connect to other spokes that are attached to the same hub.

  Network Connectivity Center retains details about deactivated
  spokes so that they can be reactivated later.
  