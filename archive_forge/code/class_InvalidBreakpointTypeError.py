from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
class InvalidBreakpointTypeError(DebugError):

    def __init__(self, type_name):
        super(InvalidBreakpointTypeError, self).__init__('{0} is not a valid breakpoint type'.format(type_name.capitalize()))