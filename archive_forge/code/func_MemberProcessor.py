from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
def MemberProcessor(member):
    """Validates and parses a service account from member string.

  Expects string.

  Args:
    member: string in format of 'serviceAccount:<value>'.

  Raises:
    MemberParseError: if string is not in valid format 'serviceAccount:<value>',
    raises exception MemberParseError.

  Returns:
    string: Returns <value> part from 'serviceAccount:<value>'.
  """
    member_array = member.split(':')
    if len(member_array) == 2 and member_array[0] == MEMBER_PREFIX:
        return member_array[1]
    else:
        raise MemberParseError(MEMBER_PARSE_ERROR.format(member))