from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ids import ids_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddTrafficLogsArg(parser, help_text='Whether to enable traffic logs on the endpoint. Enabling traffic logs can generate a large number of logs which can increase costs in Cloud Logging.'):
    parser.add_argument('--enable-traffic-logs', dest='enable_traffic_logs', required=False, default=False, help=help_text, action='store_true')