from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.health_checks import flags as health_checks_flags
def _InitialDelayValidator(value):
    duration_parser = arg_parsers.Duration(parsed_unit='s')
    parsed_value = duration_parser(value)
    if parsed_value > _MAX_INITIAL_DELAY_DURATION:
        raise arg_parsers.ArgumentTypeError('The value of initial delay must be between 0 and {max_value}'.format(max_value=_MAX_INITIAL_DELAY_DURATION_HUMAN_READABLE))
    return parsed_value