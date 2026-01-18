from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
import six.moves.urllib.parse
def AddRegionConfigArg(name, parser, repeatable=True, required=True):
    capacity_validator = arg_parsers.RegexpValidator('[0-9]+', 'capacity must be a number')
    repeatable_help = '\nThis is a repeatable flag.' if repeatable else ''
    parser.add_argument(name, help=_REGION_CONFIG_ARG_HELP_TEXT + repeatable_help, type=arg_parsers.ArgDict(spec={'region': RegionValidator, 'capacity': capacity_validator, 'enable_autoscaling': arg_parsers.ArgBoolean(), 'autoscaling_buffer': arg_parsers.BoundedInt(lower_bound=1), 'autoscaling_min_capacity': arg_parsers.BoundedInt(lower_bound=1)}, required_keys=['region', 'capacity']), required=required, action='append')