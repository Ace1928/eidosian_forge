from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.service_extensions import util
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddLogConfigFlag(parser):
    parser.add_argument('--log-config', action='append', type=util.LogConfig(), required=False, metavar='LOG_CONFIG', help=textwrap.dedent('        Logging options for the activity performed by this plugin.\n        The following options can be set:\n        * `enable`: whether to enable logging. If `log-config` flag is set,\n          `enable` option is required.\n\n        * `sample-rate`: configures the sampling rate of activity logs, where\n          `1.0` means all logged activity is reported and `0.0` means no\n          activity is reported. The default value is `1.0`, and the value of\n          the field must be in range `0` to `1` (inclusive).\n\n        * `min-log-level`: specifies the lowest level of the logs that\n          should be exported to Cloud Logging. The default value is `INFO`.\n\n        Example usage:\n        `--log-config=enable=True,sample-rate=0.5,min-log-level=INFO\n        --log_config=enable=False`\n        '))