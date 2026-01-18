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
Accept a spoke into a hub.

  Accept a proposed or previously rejected VPC spoke. By accepting a spoke,
  you permit connectivity between the associated VPC network
  and other VPC networks that are attached to the same hub.
  