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
def AddTriggerFlagGroup(parser):
    """Add arguments specifying functions trigger to the parser.

  Args:
    parser: the argparse parser for the command.
  """
    trigger_group = parser.add_mutually_exclusive_group(help="      If you don't specify a trigger when deploying an update to an existing\n      function it will keep its current trigger. You must specify one of the\n      following when deploying a new function:\n      - `--trigger-topic`,\n      - `--trigger-bucket`,\n      - `--trigger-http`,\n      - `--trigger-event` AND `--trigger-resource`,\n      - `--trigger-event-filters` and optionally `--trigger-event-filters-path-pattern`.\n      ")
    trigger_group.add_argument('--trigger-topic', help='Name of Pub/Sub topic. Every message published in this topic will trigger function execution with message contents passed as input data. Note that this flag does not accept the format of projects/PROJECT_ID/topics/TOPIC_ID. Use this flag to specify the final element TOPIC_ID. The PROJECT_ID will be read from the active configuration.', type=api_util.ValidatePubsubTopicNameOrRaise)
    trigger_group.add_argument('--trigger-bucket', help='Google Cloud Storage bucket name. Trigger the function when an object is created or overwritten in the specified Cloud Storage bucket.', type=api_util.ValidateAndStandarizeBucketUriOrRaise)
    trigger_group.add_argument('--trigger-http', action='store_true', help='      Function will be assigned an endpoint, which you can view by using\n      the `describe` command. Any HTTP request (of a supported type) to the\n      endpoint will trigger function execution. Supported HTTP request\n      types are: POST, PUT, GET, DELETE, and OPTIONS.')
    eventarc_trigger_group = trigger_group.add_argument_group()
    concept_parsers.ConceptParser([presentation_specs.ResourcePresentationSpec('--trigger-channel', eventarc_flags.ChannelResourceSpec(), '              The channel to use in the trigger for third-party event sources.\n              This is only relevant when `--gen2` is provided.', flag_name_overrides={'location': ''}, group=eventarc_trigger_group, hidden=True)], command_level_fallthroughs={'--trigger-channel.location': ['--trigger-location']}).AddToParser(parser)
    eventarc_trigger_group.add_argument('--trigger-event-filters', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, metavar='ATTRIBUTE=VALUE', help="      The Eventarc matching criteria for the trigger. The criteria can be\n      specified either as a single comma-separated argument or as multiple\n      arguments. The filters must include the ``type'' attribute, as well as any\n      other attributes that are expected for the chosen type. This is only\n      relevant when `--gen2` is provided.\n      ")
    eventarc_trigger_group.add_argument('--trigger-event-filters-path-pattern', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, metavar='ATTRIBUTE=PATH_PATTERN', help="      The Eventarc matching criteria for the trigger in path pattern format.\n      The criteria can be specified as a single comma-separated argument or as\n      multiple arguments. This is only relevant when `--gen2` is provided.\n\n      The provided attribute/value pair will be used with the\n      `match-path-pattern` operator to configure the trigger, see\n      https://cloud.google.com/eventarc/docs/reference/rest/v1/projects.locations.triggers#eventfilter\n      and https://cloud.google.com/eventarc/docs/path-patterns for more details about on\n      how to construct path patterns.\n\n      For example, to filter on events for Compute Engine VMs in a given zone:\n      `--trigger-event-filters-path-pattern=resourceName='/projects/*/zones/us-central1-a/instances/*'")
    trigger_provider_spec_group = trigger_group.add_argument_group()
    trigger_provider_spec_group.add_argument('--trigger-event', metavar='EVENT_TYPE', help='Specifies which action should trigger the function. For a list of acceptable values, call `gcloud functions event-types list`.')
    trigger_provider_spec_group.add_argument('--trigger-resource', metavar='RESOURCE', help='Specifies which resource from `--trigger-event` is being observed. E.g. if `--trigger-event` is  `providers/cloud.storage/eventTypes/object.change`, `--trigger-resource` must be a bucket name. For a list of expected resources, call `gcloud functions event-types list`.')