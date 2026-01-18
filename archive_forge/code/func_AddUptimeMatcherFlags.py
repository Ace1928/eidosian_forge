from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def AddUptimeMatcherFlags(parser):
    """Adds uptime check matcher flags to the parser."""
    uptime_matcher_group = parser.add_group(help='Uptime check matcher settings.')
    uptime_matcher_group.add_argument('--matcher-content', required=True, type=str, help='String, regex or JSON content to match.')
    uptime_matcher_group.add_argument('--matcher-type', choices=UPTIME_MATCHER_TYPES, help='The type of content matcher that is applied to the server output, defaults to\n        `contains-string`.')
    uptime_json_matcher_group = uptime_matcher_group.add_group(help='Uptime check matcher settings for JSON responses.')
    uptime_json_matcher_group.add_argument('--json-path', type=str, required=True, help='JSONPath within the response output pointing to the expected content to match.\n            Only used if `--matcher-type` is `matches-json-path` or `not-matches-json-path`.')
    uptime_json_matcher_group.add_argument('--json-path-matcher-type', choices=UPTIME_JSON_MATCHER_TYPES, help='The type of JSONPath match that is applied to the JSON output, defaults to\n            `exact-match`.\n            Only used if `--matcher-type` is `matches-json-path` or `not-matches-json-path`.')