from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddDatabaseFlagsFlag(parser):
    """Adds a --database-flags flag to the given parser."""
    help_text = "    Comma-separated list of database flags to set on the AlloyDB primary\n    instance. Use an equals sign to separate the flag name and value. Flags\n    without values, like skip_grant_tables, can be written out without a value,\n    e.g., `skip_grant_tables=`. Use on/off values for booleans. View AlloyDB's\n    documentation for allowed flags (e.g., `--database-flags\n    max_allowed_packet=55555,skip_grant_tables=,log_output=1`).\n  "
    parser.add_argument('--database-flags', type=arg_parsers.ArgDict(), metavar='FLAG=VALUE', help=help_text)