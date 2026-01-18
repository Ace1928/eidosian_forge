from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.health_checks import exceptions
from googlecloudsdk.command_lib.compute.http_health_checks import flags
from googlecloudsdk.core import log
def CreateReference(self, resources, args):
    return self.HTTP_HEALTH_CHECKS_ARG.ResolveAsResource(args, resources)