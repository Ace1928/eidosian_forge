from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
class BreakpointNotFoundError(DebugError):

    def __init__(self, breakpoint_ids, type_name):
        super(BreakpointNotFoundError, self).__init__('{0} ID not found: {1}'.format(type_name.capitalize(), ', '.join(breakpoint_ids)))