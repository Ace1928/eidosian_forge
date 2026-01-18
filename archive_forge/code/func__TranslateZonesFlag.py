from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import filter_scope_rewriter
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.resource import resource_projector
import six
def _TranslateZonesFlag(args, resources, message=None):
    """Translates --zones flag into filter expression and scope set."""
    default = args.filter
    scope_set = ZoneSet([resources.Parse(z, params={'project': properties.VALUES.core.project.GetOrFail}, collection='compute.zones') for z in args.zones])
    filter_arg = '({}) AND '.format(args.filter) if args.filter else ''
    zone_regexp = ' '.join([zone for zone in args.zones])
    zone_arg = '(zone :({}))'.format(zone_regexp)
    args.filter = filter_arg + zone_arg or default
    args.filter, filter_expr = flags.RewriteFilter(args, message=message)
    return (filter_expr, scope_set)