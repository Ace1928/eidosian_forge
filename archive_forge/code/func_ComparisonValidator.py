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
def ComparisonValidator(if_value):
    """Validates and returns the comparator and value."""
    if if_value.lower() == 'absent':
        return (None, None)
    if len(if_value) < 2:
        raise exceptions.BadArgumentException('--if', 'Invalid value for flag.')
    comparator_part = if_value[0]
    threshold_part = if_value[1:]
    try:
        comparator = COMPARISON_TO_ENUM[comparator_part]
        threshold_value = float(threshold_part)
        if comparator not in ['COMPARISON_LT', 'COMPARISON_GT']:
            raise exceptions.BadArgumentException('--if', 'Comparator must be < or >.')
        return (comparator, threshold_value)
    except KeyError:
        raise exceptions.BadArgumentException('--if', 'Comparator must be < or >.')
    except ValueError:
        raise exceptions.BadArgumentException('--if', 'Threshold not a value float.')