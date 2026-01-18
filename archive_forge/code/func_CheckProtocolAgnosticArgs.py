from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def CheckProtocolAgnosticArgs(args):
    """Raises exception if any protocol-agnostic args are invalid."""
    if args.check_interval is not None and (args.check_interval < CHECK_INTERVAL_LOWER_BOUND_SEC or args.check_interval > CHECK_INTERVAL_UPPER_BOUND_SEC):
        raise hc_exceptions.ArgumentError('[--check-interval] must not be less than {0} second or greater than {1} seconds; received [{2}] seconds.'.format(CHECK_INTERVAL_LOWER_BOUND_SEC, CHECK_INTERVAL_UPPER_BOUND_SEC, args.check_interval))
    if args.timeout is not None and (args.timeout < TIMEOUT_LOWER_BOUND_SEC or args.timeout > TIMEOUT_UPPER_BOUND_SEC):
        raise hc_exceptions.ArgumentError('[--timeout] must not be less than {0} second or greater than {1} seconds; received: [{2}] seconds.'.format(TIMEOUT_LOWER_BOUND_SEC, TIMEOUT_UPPER_BOUND_SEC, args.timeout))
    if args.healthy_threshold is not None and (args.healthy_threshold < THRESHOLD_LOWER_BOUND or args.healthy_threshold > THRESHOLD_UPPER_BOUND):
        raise hc_exceptions.ArgumentError('[--healthy-threshold] must be an integer between {0} and {1}, inclusive; received: [{2}].'.format(THRESHOLD_LOWER_BOUND, THRESHOLD_UPPER_BOUND, args.healthy_threshold))
    if args.unhealthy_threshold is not None and (args.unhealthy_threshold < THRESHOLD_LOWER_BOUND or args.unhealthy_threshold > THRESHOLD_UPPER_BOUND):
        raise hc_exceptions.ArgumentError('[--unhealthy-threshold] must be an integer between {0} and {1}, inclusive; received [{2}].'.format(THRESHOLD_LOWER_BOUND, THRESHOLD_UPPER_BOUND, args.unhealthy_threshold))