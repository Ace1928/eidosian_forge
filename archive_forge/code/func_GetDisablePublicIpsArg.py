from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetDisablePublicIpsArg(required=False):
    return base.Argument('--disable-public-ips', required=required, default=None, action=actions.StoreBooleanProperty(properties.VALUES.datapipelines.disable_public_ips), help='Specifies that Cloud Dataflow workers must not use public IP addresses by default.')