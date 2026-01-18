from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
def ConvertPriorityToInt(priority):
    try:
        int_priority = int(priority)
    except ValueError:
        raise calliope_exceptions.InvalidArgumentException('priority', 'priority must be a valid non-negative integer.')
    if int_priority < 0:
        raise calliope_exceptions.InvalidArgumentException('priority', 'priority must be a valid non-negative integer.')
    return int_priority