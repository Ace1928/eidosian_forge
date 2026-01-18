from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.networking.routers import routers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.edge_cloud.networking import resource_args
from googlecloudsdk.command_lib.edge_cloud.networking.routers import flags as routers_flags
from googlecloudsdk.core import log
Add an interface to a Distributed Cloud Edge Network router.

  *{command}* is used to add an interface to a Distributed Cloud Edge Network
  router.
  