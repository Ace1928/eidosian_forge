from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from argcomplete.completers import DirectoriesCompleter
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.eventarc import flags as eventarc_flags
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def AddVPCConnectorMutexGroup(parser):
    """Add flag for specifying VPC connector to the parser."""
    mutex_group = parser.add_group(mutex=True)
    resource = presentation_specs.ResourcePresentationSpec('--vpc-connector', GetVpcConnectorResourceSpec(), '        The VPC Access connector that the function can connect to. It can be\n        either the fully-qualified URI, or the short name of the VPC Access\n        connector resource. If the short name is used, the connector must\n        belong to the same project. The format of this field is either\n        `projects/${PROJECT}/locations/${LOCATION}/connectors/${CONNECTOR}`\n        or `${CONNECTOR}`, where `${CONNECTOR}` is the short name of the VPC\n        Access connector.\n      ', group=mutex_group, flag_name_overrides={'region': ''})
    concept_parsers.ConceptParser([resource], command_level_fallthroughs={'--vpc-connector.region': ['--region']}).AddToParser(parser)
    mutex_group.add_argument('--clear-vpc-connector', action='store_true', help='        Clears the VPC connector field.\n      ')