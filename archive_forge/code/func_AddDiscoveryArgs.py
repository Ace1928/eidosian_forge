from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.dataplex import parsers as dataplex_parsers
from googlecloudsdk.command_lib.util.args import labels_util
def AddDiscoveryArgs(parser):
    """Adds Discovery Args to parser."""
    discovery_spec = parser.add_group(help='Settings to manage the metadata discovery and publishing.')
    discovery_spec.add_argument('--discovery-enabled', action=arg_parsers.StoreTrueFalseAction, help='Whether discovery is enabled.')
    discovery_spec.add_argument('--discovery-include-patterns', default=[], type=arg_parsers.ArgList(), metavar='INCLUDE_PATTERNS', help='The list of patterns to apply for selecting data to include\n        during discovery if only a subset of the data should considered. For\n        Cloud Storage bucket assets, these are interpreted as glob patterns\n        used to match object names. For BigQuery dataset assets, these are\n        interpreted as patterns to match table names.')
    discovery_spec.add_argument('--discovery-exclude-patterns', default=[], type=arg_parsers.ArgList(), metavar='EXCLUDE_PATTERNS', help='The list of patterns to apply for selecting data to exclude\n        during discovery. For Cloud Storage bucket assets, these are interpreted\n        as glob patterns used to match object names. For BigQuery dataset\n        assets, these are interpreted as patterns to match table names.')
    trigger = discovery_spec.add_group(help='Determines when discovery jobs are triggered.')
    trigger.add_argument('--discovery-schedule', help='[Cron schedule](https://en.wikipedia.org/wiki/Cron) for running\n                discovery jobs periodically. Discovery jobs must be scheduled at\n                least 30 minutes apart.')
    discovery_prefix = discovery_spec.add_group(help='Describe data formats.')
    csv_option = discovery_prefix.add_group(help='Describe CSV and similar semi-structured data formats.')
    csv_option.add_argument('--csv-header-rows', type=int, help='The number of rows to interpret as header rows that should be skipped when reading data rows.')
    csv_option.add_argument('--csv-delimiter', help="The delimiter being used to separate values. This defaults to ','.")
    csv_option.add_argument('--csv-encoding', help='The character encoding of the data. The default is UTF-8.')
    csv_option.add_argument('--csv-disable-type-inference', action=arg_parsers.StoreTrueFalseAction, help='Whether to disable the inference of data type for CSV data. If true, all columns will be registered as strings.')
    json_option = discovery_prefix.add_group(help='Describe JSON data format.')
    json_option.add_argument('--json-encoding', help='The character encoding of the data. The default is UTF-8.')
    json_option.add_argument('--json-disable-type-inference', action=arg_parsers.StoreTrueFalseAction, help=' Whether to disable the inference of data type for Json data. If true, all columns will be registered as their primitive types (strings, number or boolean).')
    return discovery_spec